import json
import pathlib
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xgboost
from meshparty import meshwork
from scipy import sparse

from . import models

__all__ = [
    "make_skel_prop_df",
    "process_neuron",
    "label_spreading",
    "save_model",
    "AxonLabel",
]


def _segregation_index(n_pre_axon, n_post_axon, n_pre_dend, n_post_dend):
    split_ent = meshwork.algorithms._distribution_split_entropy(
        [[n_pre_axon, n_post_axon], [n_pre_dend, n_post_dend]]
    )
    unsplit_ent = meshwork.algorithms._distribution_split_entropy(
        [[n_pre_axon + n_pre_dend, n_post_dend + n_post_axon]]
    )
    return 1 - split_ent / unsplit_ent


def make_skel_prop_df_base(nrn: "meshwork.Meshwork") -> pd.DataFrame:
    """Make the per-vertex dataframe of skeleton properties

    Parameters
    ----------
    nrn : meshwork.Meshwork
        The meshwork object for the neuron with synapses

    Returns
    -------
    pd.DataFrame
        Feature dataframe indexed by skeleton vertex.
    """

    vol_df = nrn.anno.vol_prop.df
    vol_df["skel_idx"] = nrn.mesh_indices.to_skel_index_padded

    sk_prop_df = vol_df.groupby("skel_idx").agg(
        area_um2=pd.NamedAgg(column="area_nm2", aggfunc="sum"),
        vol_um3=pd.NamedAgg(column="size_nm3", aggfunc="sum"),
        max_dt_um=pd.NamedAgg(column="max_dt_nm", aggfunc="max"),
    )

    sk_prop_df["area_um2"] = sk_prop_df["area_um2"] / 1_000 / 1_000
    sk_prop_df["vol_um3"] = sk_prop_df["vol_um3"] / 1_000 / 1_000 / 1_000
    sk_prop_df["max_dt_um"] = sk_prop_df["max_dt_um"] / 1_000
    sk_prop_df["vol_to_area"] = sk_prop_df["vol_um3"] / sk_prop_df["area_um2"]

    pre_df = nrn.anno.pre_syn.df
    pre_df["skel_idx"] = nrn.anno.pre_syn.mesh_index.to_skel_index_padded

    post_df = nrn.anno.post_syn.df
    post_df["skel_idx"] = nrn.anno.post_syn.mesh_index.to_skel_index_padded

    sk_prop_df["syn_in"] = post_df.groupby("skel_idx").id.count()
    sk_prop_df["syn_out"] = pre_df.groupby("skel_idx").id.count()

    sk_prop_df["syn_in"] = sk_prop_df["syn_in"].fillna(0).astype(int)
    sk_prop_df["syn_out"] = sk_prop_df["syn_out"].fillna(0).astype(int)

    return sk_prop_df


def build_neiborhood_list(
    nrn: "meshwork.Meshwork",
    n_hops: int,
    direction: Literal["up", "down", "bidir"] = "down",
) -> list:
    """Build the k-hop neighorhood list for each vertex.

    Parameters
    ----------
    nrn : meshwork.Meshwork
        Neuron object
    n_hops : int
        Nops along the skeleton to include. Only integer hops are allowed currently, not distance.
    direction : Literal['up', 'down', 'bidir'], optional

    Returns
    -------
    list
        List of lists of neighbors for each vertex.
    """
    if direction == "down":
        gr = nrn.skeleton.csgraph_binary.T
    elif direction == "up":
        gr = nrn.skeleton.csgraph_binary
    elif direction == "bidir":
        gr = nrn.skeleton.csgraph_binary_undirected
    else:
        raise ValueError(
            f"Invalid direction: {direction}. Must be 'up', 'down', or 'bidir'."
        )

    neib_mat = sparse.csgraph.dijkstra(gr, limit=n_hops + 0.5)
    return [np.flatnonzero(~np.isinf(row)) for row in neib_mat]


def aggregate_sk_props(
    neibs: list,
    sk_prop_df: pd.DataFrame,
    prefix: str,
    n_hops: int,
) -> pd.DataFrame:
    """Aggregate skeleton properties for each vertex across its neighorhood."""
    sk_prop_df[f"{prefix}_area_um2"] = -1.0
    sk_prop_df[f"{prefix}_vol_um3"] = -1.0
    sk_prop_df[f"{prefix}_max_dt_um"] = -1.0
    sk_prop_df[f"{prefix}_vol_to_area"] = -1.0
    sk_prop_df[f"{prefix}_syn_in"] = -1.0
    sk_prop_df[f"{prefix}_syn_out"] = -1.0

    for ii, nn in enumerate(neibs):
        rows = sk_prop_df.loc[nn]
        relative_hops = len(rows) / (n_hops + 1)
        sk_prop_df.iloc[ii, sk_prop_df.columns.get_loc(f"{prefix}_area_um2")] = (
            rows["area_um2"].sum() / relative_hops
        )
        sk_prop_df.iloc[ii, sk_prop_df.columns.get_loc(f"{prefix}_vol_um3")] = (
            rows["vol_um3"].sum() / relative_hops
        )
        sk_prop_df.iloc[ii, sk_prop_df.columns.get_loc(f"{prefix}_max_dt_um")] = rows[
            "max_dt_um"
        ].mean()
        sk_prop_df.iloc[ii, sk_prop_df.columns.get_loc(f"{prefix}_vol_to_area")] = (
            sk_prop_df.iloc[ii][f"{prefix}_vol_um3"]
            / sk_prop_df.iloc[ii][f"{prefix}_area_um2"]
        )
        sk_prop_df.iloc[ii, sk_prop_df.columns.get_loc(f"{prefix}_syn_in")] = (
            rows["syn_in"].sum() / relative_hops
        )
        sk_prop_df.iloc[ii, sk_prop_df.columns.get_loc(f"{prefix}_syn_out")] = (
            rows["syn_out"].sum() / relative_hops
        )
    return sk_prop_df


def add_label(sk_prop_df: pd.DataFrame, nrn: "meshwork.Meshwork") -> pd.DataFrame:
    """Add the axon label based on the is_axon annotation property"""
    sk_prop_df["is_axon"] = False
    sk_prop_df.loc[nrn.anno.is_axon.mesh_index.to_skel_index, "is_axon"] = True
    return sk_prop_df


def make_skel_prop_df(
    nrn: "meshwork.Meshwork",
    downstream_hops: int = 5,
    upstream_hops: int = 0,
    bidirectional_hops: int = 0,
    label: bool = False,
) -> pd.DataFrame:
    """Generate a dataframe of skeleton properties for each vertex, including neighborhood aggregation.

    Parameters
    ----------
    nrn : meshwork.Meshwork
        The neuron object
    downstream_hops : int, optional
        The number of hops to aggregate downstream only, by default 5
    upstream_hops : int, optional
        The number of hops to aggegrate upstream only (towards root), by default 0
    bidirectional_hops : int, optional
        Number of hops in any direction to include, by default 0
    label : bool, optional
        Add label column, by default False

    Returns
    -------
    pd.DataFrame
    """
    sk_prop_df = make_skel_prop_df_base(nrn)
    if downstream_hops > 0:
        neibs = build_neiborhood_list(nrn, n_hops=downstream_hops, direction="down")
        aggregate_sk_props(neibs, sk_prop_df, "down", n_hops=downstream_hops)
    if upstream_hops > 0:
        neibs = build_neiborhood_list(nrn, n_hops=upstream_hops, direction="up")
        aggregate_sk_props(neibs, sk_prop_df, "up", n_hops=upstream_hops)
    if bidirectional_hops > 0:
        neibs = build_neiborhood_list(nrn, n_hops=bidirectional_hops, direction="bidir")
        aggregate_sk_props(neibs, sk_prop_df, "bidir", n_hops=2 * bidirectional_hops)
    if label:
        add_label(sk_prop_df, nrn)
    return sk_prop_df


def process_neuron(
    filepath: Union[str, pathlib.Path, "meshwork.Meshwork"],
    downstream_hops: int = 5,
    upstream_hops: int = 0,
    bidirectional_hops: int = 0,
    label: bool = False,
    feature_file: Optional[str] = None,
) -> pd.DataFrame:
    """Process a neuron and save the feature dataframe to a feather file.

    Parameters
    ----------
    filepath : Union[str, pathlib.Path, meshwork.Meshwork]
        Path to the meshwork object or the meshwork object itself.
    downstream_hops : int, optional
        Number of skeleton hops to define the downstream neighborhood, by default 5
    upstream_hops : int, optional
        Number of skeleton hops to define the upstream nei, by default 0
    bidirectional_hops : int, optional
        Number of skeleton hops to define the upstream nei, by default 0
    label : bool, optional
        Add labels based on `is_axon` label, by default False
    feature_file : Optional[str], optional
        String or path to save output file, by default None

    Returns
    -------
    pd.DataFrame
        Feature dataframe
    """
    if isinstance(filepath, meshwork.Meshwork):
        nrn = filepath
    else:
        nrn = meshwork.load_meshwork(filepath)
    sk_prop_df = make_skel_prop_df(
        nrn,
        downstream_hops=downstream_hops,
        upstream_hops=upstream_hops,
        bidirectional_hops=bidirectional_hops,
        label=label,
    )
    if feature_file is not None:
        sk_prop_df.to_feather(feature_file)
    return sk_prop_df


def _sk_laplacian_offset(
    nrn: "meshwork.Meshwork",
) -> sparse.spmatrix:
    """Compute the degree-normalized adjacency matrix part of the Laplacian matrix.

    Parameters
    ----------
    nrn : meshwork.Meshwork
        Neuron object

    Returns
    -------
    sparse.spmatrix
        Degree-normalized adjacency matrix in sparse format.
    """
    Amat = nrn.skeleton.csgraph_binary_undirected
    deg = np.array(Amat.sum(axis=0)).squeeze()
    Dmat = sparse.diags_array(1 / np.sqrt(deg))
    Lmat = Dmat @ Amat @ Dmat
    return Lmat


def label_spreading(
    nrn: "meshwork.Meshwork",
    label: np.ndarray,
    alpha: float = 0.90,
) -> np.ndarray:
    """ "Computes a smoothed label spreading that is akin to steady-state solutions to the heat equation on the skeleton graph.

    Parameters
    ----------
    nrn : meshwork.Meshwork
        Neuron object
    label : np.ndarray
        The initial label array. Must be Nxm, where N is the number of skeleton vertices
    alpha : float, optional
        A neighborhood influence parameter between 0–1. Higher values give more influence to neighbors, by default 0.90

    Returns
    -------
    np.ndarray
        The smoothed label array
    """
    Smat = _sk_laplacian_offset(nrn)
    Imat = sparse.eye(Smat.shape[0])
    invertLap = Imat - alpha * Smat
    label = np.atleast_2d(label).reshape(Smat.shape[0], -1)
    F = sparse.linalg.spsolve(invertLap, label)
    return np.squeeze((1 - alpha) * F)


def save_model(
    model: xgboost.XGBClassifier,
    feature_columns: list,
    downstream_hops: int,
    upstream_hops: int,
    bidirectional_hops: int,
    spread_alpha: float,
    filepath: Union[str, pathlib.Path],
    model_name: str = "model",
) -> None:
    """Save a trained XGBoost model."""
    if isinstance(filepath, str):
        filepath = pathlib.Path(filepath)
    if not filepath.exists():
        filepath.mkdir(parents=True)
    modelpath = filepath / "models"
    if not modelpath.exists():
        modelpath.mkdir()
    model_filename = (
        modelpath
        / f"{model_name}_ds{downstream_hops}_us{upstream_hops}_bd{bidirectional_hops}.ubj"
    )
    model.save_model(model_filename)

    outdata = {
        "downstream_hops": downstream_hops,
        "upstream_hops": upstream_hops,
        "bidirectional_hops": bidirectional_hops,
        "feature_columns": list(feature_columns),
        "spread_alpha": spread_alpha,
        "model_file": str(model_filename.absolute()),
    }
    description_file = (
        filepath
        / f"{model_name}_ds{downstream_hops}_us{upstream_hops}_bd{bidirectional_hops}.json"
    )
    with open(description_file, "w") as f:
        json.dump(outdata, f)
    return description_file


class AxonLabel:
    def __init__(
        self,
        config: Optional[Union[dict, str]] = None,
        downstream_hops: Optional[int] = None,
        upstream_hops: Optional[int] = None,
        bidirectional_hops: Optional[int] = None,
        spread_alpha: Optional[float] = None,
        model: Optional[xgboost.XGBClassifier] = None,
        feature_columns: Optional[list] = None,
    ):
        if config is not None:
            if not isinstance(config, dict):
                config = models.load_model_config(config, dir=None)
            self._downstream_hops = config.get("downstream_hops")
            self._upstream_hops = config.get("upstream_hops")
            self._bidirectional_hops = config.get("bidirectional_hops")
            self._spread_alpha = config.get("spread_alpha")
            self._load_model(config.get("model_file"))
            self._feature_columns = config.get("feature_columns")

        elif (
            config is None
            and downstream_hops is None
            and upstream_hops is None
            and bidirectional_hops is None
            and model is None
            and spread_alpha is None
        ):
            config = models.load_model_config()
            self._downstream_hops = config.get("downstream_hops")
            self._upstream_hops = config.get("upstream_hops")
            self._bidirectional_hops = config.get("bidirectional_hops")
            self._spread_alpha = config.get("spread_alpha")
            self._load_model(config.get("model_file"))
            self._feature_columns = config.get("feature_columns")
        else:
            if (
                downstream_hops is None
                or upstream_hops is None
                or bidirectional_hops is None
                or model is None
                or spread_alpha is None
            ):
                raise ValueError(
                    "Must provide either a config dictionary or all of downstream_hops, upstream_hops, bidirectional_hops, spread_alpha and model."
                )
            self._downstream_hops = downstream_hops
            self._upstream_hops = upstream_hops
            self._bidirectional_hops = bidirectional_hops
            self._model = model
            self._feature_columns = feature_columns

    @property
    def parameters(self):
        "Feature extraction parameters"
        return {
            "downstream_hops": self._downstream_hops,
            "upstream_hops": self._upstream_hops,
            "bidirectional_hops": self._bidirectional_hops,
            "spread_alpha": self._spread_alpha,
            "feature_columns": self._feature_columns,
        }

    @property
    def model_parameters(self):
        "Model parameters"
        return self._model.get_params()

    def _extract_features(
        self,
        nrn: "meshwork.Meshwork",
        label: bool = False,
    ) -> pd.DataFrame:
        return process_neuron(
            nrn,
            downstream_hops=self._downstream_hops,
            upstream_hops=self._upstream_hops,
            bidirectional_hops=self._bidirectional_hops,
            label=label,
        )

    def predict_vertex_prob_ratio(
        self,
        nrn: "meshwork.Meshwork",
        root_is_soma: bool = False,
    ) -> np.ndarray:
        """Predict the log2 ratio of axon probability to non-axon probability for each vertex.
        Higher values are axon-like, lower values are non-axon-like.

        Parameters
        ----------
        nrn : meshwork.Meshwork
            neuron object

        Returns
        -------
        np.ndarray
            log2 ratio of axon probability to non-axon probability for each skeleton vertex.
        """
        features = self._extract_features(nrn)
        is_axon_prob = self._model.predict_proba(features[self._feature_columns].values)
        if root_is_soma:
            is_axon_prob[nrn.skeleton.root, 0] = 0.99999999
            is_axon_prob[nrn.skeleton.root, 1] = 1 - is_axon_prob[nrn.skeleton.root, 0]
        label = label_spreading(
            nrn, is_axon_prob.astype(float), alpha=self._spread_alpha
        )
        return np.log2(label[:, 1] / label[:, 0])

    def predict_segment_label(
        self,
        nrn: "meshwork.Meshwork",
        vertex_prob_ratio: Optional[np.ndarray] = None,
        root_is_soma: bool = False,
    ) -> np.ndarray:
        """Predict the log2 ratio of axon probability to non-axon probability for each vertex based on segment means.
        Higher values are axon-like, lower values are non-axon-like.

        Parameters
        ----------
        nrn : meshwork.Meshwork
            neuron object
        vertex_prob_ratio : np.ndarray, optional
            Precomputed per-vertex axon probability ratio, by default None
        root_is_soma : bool, optional
            If True, the root node is considered to be the soma and thus counts as non-axon, by default False

        Returns
        -------
        np.ndarray
            log2 ratio of axon probability to non-axon probability for each skeleton vertex.
        """
        if vertex_prob_ratio is None:
            vertex_prob_ratio = self.predict_vertex_prob_ratio(
                nrn, root_is_soma=root_is_soma
            )
        segment_mean_ratio = np.array(
            [np.mean(vertex_prob_ratio[seg]) for seg in nrn.skeleton.segments]
        )
        return segment_mean_ratio[nrn.skeleton.segment_map]

    def _find_dendrite_components(
        self,
        nrn: "meshwork.Meshwork",
        segment_label: np.ndarray,
    ) -> tuple:
        """Find the connected components of the dendrite mask based on the smoothed label spreading and segment-level vote.

        Parameters
        ----------
        nrn : meshwork.Meshwork
            neuron object
        segment_label : np.ndarray
            The smoothed label spreading to determine axon vertices
        threshold : float, optional
            The threshold for the smoothed label spreading to determine axon vertices, by default 0

        Returns
        -------
        tuple
            The connected components of the dendrite mask and the segment-level vote.
        """
        sk = nrn.skeleton
        nonroot_inds = np.setdiff1d(np.arange(sk.n_vertices), [sk.root])
        child_category = np.sign(segment_label[nonroot_inds])
        parent_category = np.sign(segment_label[sk.parent_nodes(nonroot_inds)])
        cut_points = np.flatnonzero(child_category != parent_category)

        G = sk.cut_graph(cut_points, directed=False, euclidean_weight=False)
        _, comp_labels = sparse.csgraph.connected_components(G, directed=False)

        comps = []
        comp_is_dendrite = []
        for lbl in np.unique(comp_labels):
            comps.append(np.flatnonzero(comp_labels == lbl))
            comp_is_dendrite.append(segment_label[comps[-1][0]] < 0)

        comp_is_dendrite = np.array(comp_is_dendrite)
        dend_comps = [
            c
            for c, is_dend in zip(comps, comp_is_dendrite)
            if is_dend and sk.root not in c
        ]
        dend_comp_dtr = np.array([np.min(sk.distance_to_root[c]) for c in dend_comps])
        isolated_comps = np.argsort(dend_comp_dtr)[::-1]
        return dend_comps, isolated_comps

    def _evaluate_isolated_dendrite_components(
        self,
        nrn: "meshwork.Meshwork",
        is_axon_prob: np.ndarray,
        is_axon_seg: np.ndarray,
        is_axon: np.ndarray,
        dend_comps: list,
        isolated_comps: np.ndarray,
    ):
        """Evaluate the mean segment probability and change in segregation index of flipping the
        path-to-root of isolated dendrite components to dendrite labels.
        """
        sk = nrn.skeleton
        costs = []
        gains = []
        for ii, cind in enumerate(isolated_comps):
            c = dend_comps[cind]
            ptr = sk.path_to_root(c[0])

            # Compute the cost of probability cost flipping the path-to-root isolated dendrite
            costs.append(is_axon_prob[ptr].mean() * len(ptr))

            # Compute new mask having flipped the isolated dendrite component
            is_axon_seg_option = is_axon_seg.copy()
            is_axon_seg_option[ptr] = -1
            is_axon_option = self.predict_axon_mask(
                nrn, root_compartment=True, is_axon_seg=is_axon_seg_option
            )

            # Compare segregation index of the new mask to the old mask
            pre_syn_df = nrn.anno.pre_syn.df.copy()
            post_syn_df = nrn.anno.post_syn.df.copy()
            pre_syn_df["sk_ind"] = nrn.anno.pre_syn.mesh_index.to_skel_index_padded
            post_syn_df["sk_ind"] = nrn.anno.post_syn.mesh_index.to_skel_index_padded

            pre_syn_df["is_axon"] = is_axon[pre_syn_df["sk_ind"]]
            post_syn_df["is_axon"] = is_axon[post_syn_df["sk_ind"]]

            pre_syn_df["is_axon_option"] = is_axon_option[pre_syn_df["sk_ind"]]
            post_syn_df["is_axon_option"] = is_axon_option[post_syn_df["sk_ind"]]
            gains.append(
                _segregation_index(
                    pre_syn_df.query("is_axon_option").shape[0],
                    post_syn_df.query("is_axon_option").shape[0],
                    pre_syn_df.query("~is_axon_option").shape[0],
                    post_syn_df.query("~is_axon_option").shape[0],
                )
                - _segregation_index(
                    pre_syn_df.query("is_axon").shape[0],
                    post_syn_df.query("is_axon").shape[0],
                    pre_syn_df.query("~is_axon").shape[0],
                    post_syn_df.query("~is_axon").shape[0],
                )
            )
        return costs, gains

    def _flip_isolated_dendrite_components(
        self,
        nrn: "meshwork.Meshwork",
        root_is_soma: bool = False,
        is_axon_prob: np.ndarray = None,
        is_axon_seg: np.ndarray = None,
    ):
        if is_axon_prob is None:
            is_axon_prob = self.predict_vertex_prob_ratio(nrn)
        if is_axon_seg is None:
            is_axon_seg = self.predict_segment_label(nrn)
        is_axon = self.predict_axon_mask(
            nrn,
            root_compartment=True,
            root_is_soma=root_is_soma,
            is_axon_seg=is_axon_seg,
        )

        # Find isolated dendrite components
        # TODO: Reroot to largest dendrite component if root_is_soma is False
        dend_comps, isolated_comps = self._find_dendrite_components(nrn, is_axon_seg)
        costs, gains = self._evaluate_isolated_dendrite_components(
            nrn, is_axon_prob, is_axon_seg, is_axon, dend_comps, isolated_comps
        )
        for c, g, cind in zip(costs, gains, isolated_comps):
            # If you both have a dendrite-like path to root and improve segregation, flip the path to being dendrite-like
            if c < 0 and g > 0:
                ptr = nrn.skeleton.path_to_root(dend_comps[cind][0])
                is_axon_seg[ptr] = -1
        return self.predict_axon_mask(
            nrn,
            root_compartment=True,
            root_is_soma=root_is_soma,
            is_axon_seg=is_axon_seg,
        )

    def predict_axon_mask(
        self,
        nrn: "meshwork.Meshwork",
        root_compartment: bool = True,
        threshold: float = 0,
        root_is_soma: bool = False,
        is_axon_seg: Optional[np.ndarray] = None,
        evaluate_isolated_dendrites: bool = False,
        to_mesh_index: bool = False,
    ):
        """Predict a dendrite mask based on the smoothed label spreading and segment-level vote.

        Parameters
        ----------
        nrn : meshwork.Meshwork
            neuron object
        root_compartment : bool, optional
            If True, the dendrite mask (inverse of the axon mask) only includes the same compartment
            as the root node in the dendrite mask, by default True.
        threshold : float, optional
            The threshold for the smoothed label spreading to determine axon vertices, by default 0.
        threshold : float, optional
            The threshold for the smoothed label spreading to determine axon vertices, by default 0.
        root_is_soma : bool, optional
            If True, the root node is considered to be the soma and thus counts as non-axon, by default False
        is_axon_seg : np.ndarray, optional
            Precomputed segment-level axon probability ratio, by default None
        evaluate_isolated_dendrites : bool, optional
            If True, isolated dendrite components are flipped if they improve the segregation index and are net-positive in the log-odds along the path to root, by default False

        Returns
        -------
        np.ndarray
            Boolean mask of dendrite vertices.
        """
        if is_axon_seg is None:
            seg_prob_ratio = self.predict_segment_label(nrn, root_is_soma=root_is_soma)
        else:
            seg_prob_ratio = is_axon_seg
        is_axon_base = seg_prob_ratio > threshold
        if root_compartment:
            if evaluate_isolated_dendrites:
                is_axon_base = self._flip_isolated_dendrite_components(
                    nrn, root_is_soma=root_is_soma, is_axon_seg=seg_prob_ratio
                )
            axon_inds = np.flatnonzero(is_axon_base)
            G = nrn.skeleton.cut_graph(
                axon_inds, directed=False, euclidean_weight=False
            )
            _, comp_labels = sparse.csgraph.connected_components(G, directed=False)
            in_root_comp = comp_labels == comp_labels[nrn.skeleton.root]
        else:
            in_root_comp = np.full(len(nrn.skeleton.vertices), True)
        is_axon = ~np.logical_and(~is_axon_base, in_root_comp)
        if to_mesh_index:
            is_axon_sk = nrn.SkeletonIndex(np.flatnonzero(is_axon))
            is_axon = is_axon_sk.to_mesh_mask
        return is_axon

    def _load_model(self, filepath: str):
        """Load a model from a file."""
        self._model = xgboost.XGBClassifier()
        self._model.load_model(filepath)
