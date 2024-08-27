# coding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(
        self,
        device,
        factor=1,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.device = device

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # print("corr")
        # print(corr.shape)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(
        self, correlation, d_model, n_heads, device, d_keys=None, d_values=None
    ):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.device = device

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads, device=device)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads, device=device)
        self.value_projection = nn.Linear(d_model, d_values * n_heads, device=device)
        self.out_projection = nn.Linear(d_values * n_heads, d_model, device=device)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, S_T_Weight_M=None, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values)
        out = out.view(B, L, -1)
        # if S_T_Weight_M is not None:
        #     out = torch.bmm(S_T_Weight_M, out)

        return self.out_projection(out), attn


class SpatialTemporalEnhancedLayer(nn.Module):
    """Spatial-temporal enhanced layer"""

    def __init__(
        self,
        num_users,
        num_pois,
        num_cate,
        seq_len,
        emb_dim,
        num_heads,
        dropout,
        device,
        time_slot,
    ):
        super(SpatialTemporalEnhancedLayer, self).__init__()

        self.num_users = num_users
        self.num_pois = num_pois
        self.emb_dim = emb_dim
        self.num_cate = num_cate
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.device = device

        self.pos_embeddings = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)
        self.pos_embeddings_cate = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)

        self.time_cir_embeddings = nn.Embedding(time_slot + 1, emb_dim, padding_idx=0)
        self.time_cir_embeddings_cate = nn.Embedding(
            time_slot + 1, emb_dim, padding_idx=0
        )

        self.fc_geo = nn.Linear(emb_dim, emb_dim, bias=True, device=device)
        self.fc_time = nn.Linear(emb_dim, emb_dim, bias=True, device=device)

        # self.multi_attn = nn.MultiheadAttention(
        #     emb_dim, num_heads, dropout, batch_first=True, device=device
        # )
        # self.multi_attn_cate = nn.MultiheadAttention(
        #     emb_dim, num_heads, dropout, batch_first=True, device=device
        # )
        # self.multi_attn_cross = nn.MultiheadAttention(
        #     emb_dim, num_heads, dropout, batch_first=True, device=device
        # )
        # self.multi_attn_cate_cross = nn.MultiheadAttention(
        #     emb_dim, num_heads, dropout, batch_first=True, device=device
        # )

        self.multi_ac = AutoCorrelationLayer(
            AutoCorrelation(
                device=device,
                output_attention=False,
                attention_dropout=dropout,
            ),
            emb_dim,
            num_heads,
            device,
        )
        self.multi_ac_cate = AutoCorrelationLayer(
            AutoCorrelation(
                device=device,
                output_attention=False,
                attention_dropout=dropout,
            ),
            emb_dim,
            num_heads,
            device,
        )
        self.multi_ac_cross = AutoCorrelationLayer(
            AutoCorrelation(
                device=device,
                output_attention=False,
                attention_dropout=dropout,
            ),
            emb_dim,
            num_heads,
            device,
        )
        self.multi_ac_cate_cross = AutoCorrelationLayer(
            AutoCorrelation(
                device=device,
                output_attention=False,
                attention_dropout=dropout,
            ),
            emb_dim,
            num_heads,
            device,
        )

        self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(
        self,
        G,
        G_cate,
        nodes_embeds,
        cate_embeds,
        batch_users_seqs,
        batch_users_seqs_cate,
        batch_users_seqs_masks,
        batch_users_geo_adjs,
        batch_users_time_adjs,
        batch_users_indices,
        batch_users_seqs_timeslot,
        batch_users_cate_timeslot,
    ):
        batch_size = batch_users_seqs.size(0)
        batch_users_geo_adjs = batch_users_geo_adjs.float()
        batch_users_time_adjs = batch_users_time_adjs.float()

        # generate sequence embeddings
        batch_seqs_embeds = nodes_embeds[batch_users_seqs]
        batch_seqs_cate_embeds = cate_embeds[batch_users_seqs_cate]

        # print(
        #     batch_users_seqs_cate.shape, batch_seqs_cate_embeds.shape, cate_embeds.shape
        # )

        batch_users_seqs_timeslot_poi = batch_users_seqs_timeslot - (
            self.num_users + self.num_pois
        )
        batch_users_seqs_timeslot_cate = batch_users_cate_timeslot - (
            self.num_users + self.num_cate
        )
        # print(batch_users_seqs_timeslot_poi)
        batch_seqs_timeslot_embeds = self.time_cir_embeddings(
            batch_users_seqs_timeslot_poi
        )
        batch_seqs_cate_timeslot_embeds = self.time_cir_embeddings_cate(
            batch_users_seqs_timeslot_cate
        )

        # print(batch_seqs_timeslot_embeds.shape)

        # generate position embeddings
        batch_seqs_pos = torch.arange(1, self.seq_len + 1, device=self.device)
        batch_seqs_pos = batch_seqs_pos.repeat(batch_size, 1)
        batch_seqs_pos = torch.multiply(batch_seqs_pos, batch_users_seqs_masks)
        batch_seqs_pos_embs = self.pos_embeddings(batch_seqs_pos)
        batch_seqs_pos_cate_embs = self.pos_embeddings_cate(batch_seqs_pos)

        # generate geographical embeddings
        # print(batch_users_geo_adjs.shape)

        # geo adjs is exp(-dist(x,y)^2) ,so not need to trans
        # print(batch_users_geo_adjs)
        # print(batch_seqs_embeds.shape)
        batch_seqs_geo_embeds = batch_users_geo_adjs.matmul(batch_seqs_embeds)
        # print(batch_seqs_geo_embeds.shape)
        batch_seqs_geo_embeds = torch.relu(self.fc_geo(batch_seqs_geo_embeds))
        # print(batch_seqs_geo_embeds.shape)

        # generate delta_time embeddings
        # print(batch_users_time_adjs)
        batch_seqs_delta_time_emb = batch_users_time_adjs.matmul(batch_seqs_embeds)
        # print(batch_users_time_emb.shape)
        batch_seqs_delta_time_emb = torch.relu(self.fc_time(batch_seqs_delta_time_emb))
        # print(batch_seqs_time_emb.shape)

        # multi-head attention
        batch_seqs_total_embeds = (
            batch_seqs_embeds
            + batch_seqs_pos_embs
            + batch_seqs_geo_embeds
            + batch_seqs_delta_time_emb
            + batch_seqs_timeslot_embeds
        )
        batch_seqs_cate_total_embeds = (
            batch_seqs_cate_embeds
            + batch_seqs_pos_cate_embs
            + batch_seqs_cate_timeslot_embeds
        )

        # attention
        # batch_seqs_mha, _ = self.multi_attn(
        #     batch_seqs_total_embeds, batch_seqs_total_embeds, batch_seqs_total_embeds
        # )
        # batch_seqs_cate_mha, _ = self.multi_attn_cate(
        #     batch_seqs_cate_total_embeds,
        #     batch_seqs_cate_total_embeds,
        #     batch_seqs_cate_total_embeds,
        # )

        #ac
        batch_seqs_mha, _ = self.multi_ac(
            batch_seqs_total_embeds, batch_seqs_total_embeds, batch_seqs_total_embeds
        )
        batch_seqs_cate_mha, _ = self.multi_ac_cate(
            batch_seqs_cate_total_embeds,
            batch_seqs_cate_total_embeds,
            batch_seqs_cate_total_embeds,
        )


        # cross modal
        # batch_seqs_mha_cross, _ = self.multi_attn_cross(
        #     batch_seqs_mha, batch_seqs_cate_mha, batch_seqs_cate_mha
        # )
        # batch_seqs_cate_mha_cross, _ = self.multi_attn_cate_cross(
        #     batch_seqs_cate_mha,
        #     batch_seqs_mha,
        #     batch_seqs_mha,
        # )

        batch_seqs_mha_cross, _ = self.multi_ac_cross(
            batch_seqs_mha, batch_seqs_cate_mha, batch_seqs_cate_mha
        )
        batch_seqs_cate_mha_cross, _ = self.multi_ac_cate_cross(
            batch_seqs_cate_mha,
            batch_seqs_mha,
            batch_seqs_mha,
        )

        # batch_users_embeds = torch.mean(batch_seqs_mha, dim=1)
        # batch_users_cate_embeds = torch.mean(batch_seqs_cate_mha, dim=1)
        batch_users_embeds = torch.mean(batch_seqs_mha_cross, dim=1)
        batch_users_cate_embeds = torch.mean(batch_seqs_cate_mha_cross, dim=1)
        # print(batch_seqs_mha.shape, batch_users_embeds.shape)

        nodes_embeds = nodes_embeds.clone()
        cate_embeds = cate_embeds.clone()
        nodes_embeds[batch_users_indices] = batch_users_embeds
        cate_embeds[batch_users_indices] = batch_users_cate_embeds

        # graph convolutional
        lconv_nodes_embeds = torch.sparse.mm(
            G, nodes_embeds[: self.num_users + self.num_pois]
        )
        # print("1" * 10)
        # print(G.shape, nodes_embeds[: self.num_users + self.num_pois].shape)
        lconv_nodes_cate_embeds = torch.sparse.mm(
            G_cate, cate_embeds[: self.num_users + self.num_cate]
        )
        nodes_embeds[: self.num_users + self.num_pois] = lconv_nodes_embeds
        cate_embeds[: self.num_users + self.num_cate] = lconv_nodes_cate_embeds
        # print(nodes_embeds.shape)
        return nodes_embeds, cate_embeds


class LocalSpatialTemporalGraph(nn.Module):
    """Local spatial-temporal enhanced graph neural network module"""

    def __init__(
        self,
        num_layers,
        num_users,
        num_pois,
        num_cate,
        seq_len,
        emb_dim,
        num_heads,
        dropout,
        device,
        time_slot,
    ):
        super(LocalSpatialTemporalGraph, self).__init__()

        self.num_layers = num_layers
        self.num_users = num_users
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device
        self.spatial_temporal_layers = nn.ModuleList([SpatialTemporalEnhancedLayer(
            num_users,
            num_pois,
            num_cate,
            seq_len,
            emb_dim,
            num_heads,
            dropout,
            device,
            time_slot,
        )for l in range(num_layers)])
        # self.spatial_temporal_layer = SpatialTemporalEnhancedLayer(
        #     num_users,
        #     num_pois,
        #     num_cate,
        #     seq_len,
        #     emb_dim,
        #     num_heads,
        #     dropout,
        #     device,
        #     time_slot,
        # )

    def forward(
        self,
        G,
        G_cate,
        nodes_embeds,
        cate_embeds,
        batch_users_seqs,
        batch_users_seqs_cate,
        batch_users_seqs_masks,
        batch_users_geo_adjs,
        batch_users_time_adjs,
        batch_users_indices,
        batch_users_seqs_timeslot,
        batch_users_cate_timeslot,
    ):
        nodes_embedding = [nodes_embeds]
        cate_embedding = [cate_embeds]
        for layer in range(self.num_layers):
            nodes_embeds, cate_embeds = self.spatial_temporal_layers[layer](
                G,
                G_cate,
                nodes_embeds,
                cate_embeds,
                batch_users_seqs,
                batch_users_seqs_cate,
                batch_users_seqs_masks,
                batch_users_geo_adjs,
                batch_users_time_adjs,
                batch_users_indices,
                batch_users_seqs_timeslot,
                batch_users_cate_timeslot,
            )
            nodes_embeds = F.dropout(nodes_embeds, self.dropout)
            nodes_embedding.append(nodes_embeds)
            cate_embeds = F.dropout(cate_embeds, self.dropout)
            cate_embedding.append(cate_embeds)

        nodes_embeds_tensor = torch.stack(nodes_embedding)
        final_nodes_embeds = torch.mean(nodes_embeds_tensor, dim=0)

        cate_embeds_tensor = torch.stack(cate_embedding)
        final_cate_embeds = torch.mean(cate_embeds_tensor, dim=0)

        return final_nodes_embeds, final_cate_embeds


class HypergraphConvolutionalNetwork(nn.Module):
    """Hypergraph convolutional network"""

    def __init__(self, emb_dim, num_layers, num_users, dropout, device):
        super(HypergraphConvolutionalNetwork, self).__init__()

        self.num_layers = num_layers
        self.num_users = num_users
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device

    def forward(self, x, HG):
        item_embedding = [x]
        for layer in range(self.num_layers):
            x = torch.sparse.mm(HG, x)
            x = F.dropout(x, self.dropout)
            item_embedding.append(x)

        item_embedding_tensor = torch.stack(item_embedding)
        final_item_embedding = torch.mean(item_embedding_tensor, dim=0)

        return final_item_embedding


class MSTHN(nn.Module):
    """Our proposed Multi-view Spatial-Temporal Enhanced Hypergraph Network (MSTHN)"""

    def __init__(
        self,
        num_local_layer,
        num_global_layer,
        num_users,
        num_pois,
        num_cate,
        seq_len,
        emb_dim,
        num_heads,
        dropout,
        device,
        time_slot,
    ):
        super(MSTHN, self).__init__()

        self.num_nodes = num_users + num_pois + 1
        self.num_nodes_cate = num_users + num_cate + 1
        self.num_users = num_users
        self.num_pois = num_pois
        self.num_cate = num_cate
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.padding_idx = num_users + num_pois
        self.padding_idx_cate = num_users + num_cate
        self.device = device
        self.time_slot = time_slot

        self.nodes_embeddings = nn.Embedding(
            self.num_nodes, emb_dim, padding_idx=self.padding_idx
        )

        self.node_cate_embeddings = nn.Embedding(
            self.num_nodes_cate, emb_dim, padding_idx=self.padding_idx_cate
        )

        self.pos_embeddings = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)
        self.pos_embeddings_cate = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)

        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim, device=device)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim, device=device)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False, device=device)

        self.w_1_c = nn.Linear(2 * self.emb_dim, self.emb_dim, device=device)
        self.w_2_c = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1_c = nn.Linear(self.emb_dim, self.emb_dim, device=device)
        self.glu2_c = nn.Linear(self.emb_dim, self.emb_dim, bias=False, device=device)



        # local graph and global hypergraph
        self.local_graph = LocalSpatialTemporalGraph(
            num_local_layer,
            num_users,
            num_pois,
            num_cate,
            seq_len,
            emb_dim,
            num_heads,
            dropout,
            device,
            time_slot,
        )
        self.global_hyg = HypergraphConvolutionalNetwork(
            emb_dim, num_global_layer, num_users, dropout, device
        )

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def user_temporal_pref_augmentation(
        self,
        node_embedding,
        session_len,
        reversed_sess_item,
        mask,
        pos_embeddings,
        w_1,
        w_2,
        glu1,
        glu2,
    ):
        """user temporal preference augmentation"""
        batch_size = session_len.size(0)
        seq_h = node_embedding[reversed_sess_item]
        hs = torch.div(torch.sum(seq_h, 1), session_len.unsqueeze(-1))

        batch_seqs_pos = torch.arange(1, self.seq_len + 1, device=self.device)
        batch_seqs_pos = batch_seqs_pos.repeat(batch_size, 1)
        batch_seqs_pos = torch.multiply(batch_seqs_pos, mask)
        pos_emb = pos_embeddings(batch_seqs_pos)

        hs = hs.unsqueeze(1).repeat(1, self.seq_len, 1)
        nh = w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(glu1(nh) + glu2(hs))
        beta = torch.matmul(nh, w_2)
        select = torch.sum(beta * seq_h, 1)

        return select

    def timestamp2time_slot(self, batch_users_seqs_time, time_slot, padding_idx):
        batch_users_seqs_time_temp = torch.full(
            batch_users_seqs_time.shape,
            padding_idx,
            dtype=torch.int64,
            device=self.device,
        )

        # 将非零值进行计算，并将计算结果存储在 result 张量中
        non_zero_indices = torch.nonzero(batch_users_seqs_time)
        non_zero_values = batch_users_seqs_time[
            non_zero_indices[:, 0], non_zero_indices[:, 1]
        ]
        computed_values = (non_zero_values - 1) % time_slot + 1 + padding_idx
        batch_users_seqs_time_temp[
            non_zero_indices[:, 0], non_zero_indices[:, 1]
        ] = computed_values

        return batch_users_seqs_time_temp

    def forward(
        self,
        G,
        G_cate,
        HG,
        batch_users_seqs,
        batch_users_seqs_masks,
        batch_users_geo_adjs,
        batch_users_time_adjs,
        batch_users_indices,
        batch_seqs_lens,
        batch_users_rev_seqs,
    ):
        batch_users_seqs_poi = batch_users_seqs[:, :, 0].to(torch.int64)
        batch_users_seqs_cate = batch_users_seqs[:, :, 2].to(torch.int64)

        batch_users_seqs_time = batch_users_seqs[:, :, 1].to(torch.int64)

        batch_users_rev_seqs_poi = batch_users_rev_seqs[:, :, 0].to(torch.int64)
        batch_users_rev_seqs_cate = batch_users_rev_seqs[:, :, 2].to(torch.int64)
        # batch_users_rev_seqs_time = batch_users_rev_seqs[:, :, 1].to(torch.int64)
        # print(batch_users_rev_seqs[0:1, :, 1].shape)

        # transform timestamp to time slot
        batch_users_seqs_timeslot = self.timestamp2time_slot(
            batch_users_seqs_time, self.time_slot, self.padding_idx
        )
        batch_users_cate_timeslot = self.timestamp2time_slot(
            batch_users_seqs_time, self.time_slot, self.padding_idx_cate
        )
        # print(batch_users_seqs_timeslot[0])

        # batch_users_rev_seqs_time = (batch_users_rev_seqs_time - 1) % self.time_slot + 1
        nodes_embeds = self.nodes_embeddings.weight
        cate_embeds = self.node_cate_embeddings.weight
        # print(nodes_embeds[self.num_users : self.padding_idx, :].shape)
        local_nodes_embs, local_cate_embs = self.local_graph(
            G,
            G_cate,
            nodes_embeds,
            cate_embeds,
            batch_users_seqs_poi,
            batch_users_seqs_cate,
            batch_users_seqs_masks,
            batch_users_geo_adjs,
            batch_users_time_adjs,
            batch_users_indices,
            batch_users_seqs_timeslot,
            batch_users_cate_timeslot,
        )
        # poi prediction
        local_batch_users_embs = local_nodes_embs[batch_users_indices]
        local_pois_embs = local_nodes_embs[self.num_users : self.padding_idx, :]

        global_pois_embs = self.global_hyg(
            nodes_embeds[self.num_users : self.padding_idx, :], HG
        )

        pois_embs = local_pois_embs + global_pois_embs
        # pois_embs = local_pois_embs
        fusion_nodes_embs = torch.cat(
            [local_nodes_embs[: self.num_users], pois_embs], dim=0
        )
        # print(fusion_nodes_embs.shape)
        fusion_nodes_embs = torch.cat(
            [
                fusion_nodes_embs,
                torch.zeros(size=(1, self.emb_dim), device=self.device),
            ],
            dim=0,
        )

        batch_users_embs = self.user_temporal_pref_augmentation(
            fusion_nodes_embs,
            batch_seqs_lens,
            batch_users_rev_seqs_poi,
            batch_users_seqs_masks,
            self.pos_embeddings,
            self.w_1,
            self.w_2,
            self.glu1,
            self.glu2,
        )
        batch_users_embs = batch_users_embs + local_batch_users_embs


        # cate prediction
        local_batch_users_cate_embs = local_cate_embs[batch_users_indices]
        cate_embs = local_cate_embs[self.num_users : self.padding_idx_cate, :]

        fusion_nodes_cate_embs = local_cate_embs[: self.padding_idx_cate, :]
        # # print(fusion_nodes_embs.shape)
        fusion_cate_embs = torch.cat(
            [
                fusion_nodes_cate_embs,
                torch.zeros(size=(1, self.emb_dim), device=self.device),
            ],
            dim=0,
        )
        batch_users_cate_embs = self.user_temporal_pref_augmentation(
            fusion_cate_embs,
            batch_seqs_lens,
            batch_users_rev_seqs_cate,
            batch_users_seqs_masks,
            self.pos_embeddings_cate,
            self.w_1_c,
            self.w_2_c,
            self.glu1_c,
            self.glu2_c,
        )
        batch_users_cate_embs = batch_users_cate_embs + local_batch_users_cate_embs

        prediction_poi = torch.matmul(batch_users_embs, pois_embs.t())

        prediction_cate = torch.matmul(batch_users_cate_embs, cate_embs.t())
        return prediction_poi, prediction_cate
