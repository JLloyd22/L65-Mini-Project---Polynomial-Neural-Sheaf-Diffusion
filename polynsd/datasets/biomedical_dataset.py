#  Copyright (c) 2024. Luke Braithwaite
#  License: MIT

"""Biomedical Network Datasets for Link Prediction.

This module provides dataset classes for various biomedical networks used in
drug-disease association (DDA), drug-drug interaction (DDI), drug-target interaction (DTI),
and protein-protein interaction (PPI) prediction tasks.

Datasets are sourced from:
- BioNEV: https://github.com/xiangyue9607/BioNEV
- BioERP: https://github.com/pengsl-lab/BioERP
"""

import os
import urllib.request
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from collections import defaultdict
import random
from .link_pred import LinkPredBase, DATA_DIR


class BiomedicalDataset(InMemoryDataset):
    """PyTorch Geometric Dataset wrapper for biomedical networks.

    This class handles downloading, processing, and caching of biomedical network data
    from GitHub repositories.
    """

    _base_urls = {
        # BioERP datasets (complex biomedical networks)
        "NeoDTI-Net": "https://raw.githubusercontent.com/pengsl-lab/BioERP/master/data/NeoDTI-Net/",
        "deepDR-Net": "https://raw.githubusercontent.com/pengsl-lab/BioERP/master/data/deepDR-Net/",
        # BioNEV datasets (single biomedical networks)
        "CTD_DDA": "https://raw.githubusercontent.com/xiangyue9607/BioNEV/master/data/CTD_DDA/",
        "NDFRT_DDA": "https://raw.githubusercontent.com/xiangyue9607/BioNEV/master/data/NDFRT_DDA/",
        "DrugBank_DDI": "https://raw.githubusercontent.com/xiangyue9607/BioNEV/master/data/DrugBank_DDI/",
        "STRING_PPI": "https://raw.githubusercontent.com/xiangyue9607/BioNEV/master/data/STRING_PPI/",
    }

    _file_lists = {
        "NeoDTI-Net": [
            "drug_dict.txt",
            "protein_dict.txt",
            "disease_dict.txt",
            "se_dict.txt",
            "drug_drug.txt",
            "drug_protein.txt",
            "drug_disease.txt",
            "drug_se.txt",
            "protein_protein.txt",
            "protein_disease.txt",
        ],
        "deepDR-Net": [
            "drugDisease.txt",
            "drugdrug.txt",
            "drugProtein.txt",
            "drugsideEffect.txt",
        ],
        "CTD_DDA": ["node_list.txt", "CTD_DDA.edgelist"],
        "NDFRT_DDA": ["node_list.txt", "NDFRT_DDA.edgelist"],
        "DrugBank_DDI": ["node_list.txt", "DrugBank_DDI.edgelist"],
        "STRING_PPI": ["node_list.txt", "STRING_PPI.edgelist"],
    }

    def __init__(
        self,
        name,
        root="./data",
        transform=None,
        pre_transform=None,
        subsample_ratio=1.0,
        node_feature_type="constant",
    ):
        assert name in self._base_urls.keys(), f"Dataset {name} not supported"
        assert node_feature_type in ["constant", "degree"], (
            f"node_feature_type must be 'constant' or 'degree', got {node_feature_type}"
        )

        self.name = name
        self.url = self._base_urls[name]
        self.subsample_ratio = subsample_ratio
        self.node_feature_type = node_feature_type
        super(BiomedicalDataset, self).__init__(
            root=os.path.join(root, name),
            transform=transform,
            pre_transform=pre_transform,
        )
        # Load with weights_only=False to support PyTorch Geometric objects
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        """Return list of raw data files."""
        return self._file_lists[self.name]

    @property
    def processed_file_names(self):
        """Return list of processed data files."""
        return ["data.pt"]

    def download(self):
        """Download raw data files from GitHub repositories or zip archives."""
        import zipfile

        for filename in self.raw_file_names:
            file_path = os.path.join(self.raw_dir, filename)

            if not os.path.exists(file_path):
                # For zip files, download directly (URL already has filename)
                if filename.endswith(".zip"):
                    file_url = self.url
                else:
                    file_url = self.url + filename

                try:
                    print(f"Downloading {filename}...")
                    urllib.request.urlretrieve(file_url, file_path)

                    # Extract zip files
                    if filename.endswith(".zip"):
                        print(f"Extracting {filename}...")
                        with zipfile.ZipFile(file_path, "r") as zip_ref:
                            zip_ref.extractall(self.raw_dir)
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    print(f"Please manually download from {file_url}")
                    print(f"and save to {file_path}")

    def _compute_degree_features(self, num_nodes, edge_list):
        """Compute node degree features (in-degree and out-degree).

        We compute degree-based features instead of using constant features
        (like T.Constant() which gives all nodes x=[1.0]) because:


        Degree features capture node importance: Nodes with high connectivity
        (hubs) get different features than peripheral nodes, allowing the model
        to learn node-specific patterns.

        Log normalization prevents extreme values: log(degree+1) keeps features
        in a reasonable range while preserving relative differences.

        Example: A drug node connected to 100 diseases gets features [log(101), log(X)]
                 while a drug connected to 1 disease gets [log(2), log(Y)]
                 This helps the model distinguish important vs rare entities.

        Args:
            num_nodes: Total number of nodes
            edge_list: List of (src, dst) tuples

        Returns:
            torch.Tensor: [num_nodes, 2] with [log(in_degree+1), log(out_degree+1)]
        """
        in_degree = torch.zeros(num_nodes, dtype=torch.float)
        out_degree = torch.zeros(num_nodes, dtype=torch.float)

        # Count incoming and outgoing edges for each node
        for src, dst in edge_list:
            out_degree[src] += 1  # src node has one more outgoing edge
            in_degree[dst] += 1  # dst node has one more incoming edge

        # Apply log(degree + 1) normalization:
        # - +1 handles nodes with degree=0 (log(0) is undefined)
        # - log() compresses large degree values (e.g., log(1000)≈7 vs 1000)
        in_degree = torch.log(in_degree + 1).unsqueeze(1)
        out_degree = torch.log(out_degree + 1).unsqueeze(1)

        # Return [num_nodes, 2] tensor where each row is [log(in_deg+1), log(out_deg+1)]
        return torch.cat([in_degree, out_degree], dim=1)

    def _load_node_dict(self, filepath):
        """Load node dictionary mapping IDs to names."""
        node_dict = {}
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    node_id, node_name = parts[0], " ".join(parts[1:])
                    # Try to convert to int, otherwise keep as string
                    try:
                        node_id = int(node_id)
                    except ValueError:
                        pass
                    node_dict[node_id] = node_name
        return node_dict

    def _load_edges(self, filepath, subsample_ratio=1.0):
        """Load edge list from file.
        Handles both edge list format (two integers per line)
        and adjacency matrix format (space or tab-separated 0s and 1s).

        Args:
            filepath: Path to edge file
            subsample_ratio: Fraction of edges to keep (for memory efficiency with large adjacency matrices)
        """
        edges = []
        with open(filepath, "r") as f:
            lines = f.readlines()

            # Detect delimiter: check if first line has tabs or spaces
            first_line = lines[0].strip()
            if "\t" in first_line:
                delimiter = "\t"
                first_line_parts = first_line.split("\t")
            else:
                delimiter = None  # split() with no args splits on any whitespace
                first_line_parts = first_line.split()

            # Check if first line is adjacency matrix format (many values)
            if len(first_line_parts) > 10:  # Likely an adjacency matrix
                # Parse as adjacency matrix
                for i, line in enumerate(lines):
                    if delimiter:
                        values = list(map(int, line.strip().split(delimiter)))
                    else:
                        values = list(map(int, line.strip().split()))
                    for j, val in enumerate(values):
                        if val == 1:  # Edge exists
                            # Subsample edges if requested (for memory efficiency)
                            if (
                                subsample_ratio >= 1.0
                                or random.random() < subsample_ratio
                            ):
                                edges.append((i, j))
            else:
                # Parse as edge list
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            src, dst = int(parts[0]), int(parts[1])
                            edges.append((src, dst))
                        except ValueError:
                            continue
        return edges

    def process(self):
        """Process raw data to create PyG HeteroData graph."""
        if self.name == "NeoDTI-Net":
            data = self._process_neodti_net()
        elif self.name == "deepDR-Net":
            data = self._process_deepdr_net()
        elif self.name in ["CTD_DDA", "NDFRT_DDA"]:
            data = self._process_dda_network()
        elif self.name == "DrugBank_DDI":
            data = self._process_ddi_network()
        elif self.name == "STRING_PPI":
            data = self._process_ppi_network()

        torch.save(self.collate([data]), self.processed_paths[0])

    def _process_neodti_net(self):
        """Process NeoDTI-Net dataset."""
        data = HeteroData()

        # Load edge lists with subsampling to reduce edge count (configurable via subsample_ratio)
        # NeoDTI has dense adjacency matrices causing memory issues - subsample for tractability
        drug_drug = self._load_edges(
            os.path.join(self.raw_dir, "drug_drug.txt"),
            subsample_ratio=self.subsample_ratio,
        )
        drug_protein = self._load_edges(
            os.path.join(self.raw_dir, "drug_protein.txt"),
            subsample_ratio=self.subsample_ratio,
        )
        drug_disease = self._load_edges(
            os.path.join(self.raw_dir, "drug_disease.txt"),
            subsample_ratio=self.subsample_ratio,
        )
        drug_se = self._load_edges(
            os.path.join(self.raw_dir, "drug_se.txt"),
            subsample_ratio=self.subsample_ratio,
        )
        protein_protein = self._load_edges(
            os.path.join(self.raw_dir, "protein_protein.txt"),
            subsample_ratio=self.subsample_ratio,
        )
        protein_disease = self._load_edges(
            os.path.join(self.raw_dir, "protein_disease.txt"),
            subsample_ratio=self.subsample_ratio,
        )

        # Determine actual node counts from edge indices (max node ID + 1)
        num_drugs = (
            max(
                [max(s, d) for s, d in drug_drug]
                + [s for s, d in drug_protein]
                + [s for s, d in drug_disease]
                + [s for s, d in drug_se],
                default=-1,
            )
            + 1
        )
        num_proteins = (
            max(
                [d for s, d in drug_protein]
                + [max(s, d) for s, d in protein_protein]
                + [s for s, d in protein_disease],
                default=-1,
            )
            + 1
        )
        num_diseases = (
            max(
                [d for s, d in drug_disease] + [d for s, d in protein_disease],
                default=-1,
            )
            + 1
        )
        num_sideeffects = max([d for s, d in drug_se], default=-1) + 1

        # Set number of nodes for each type based on actual adjacency matrix dimensions
        data["drug"].num_nodes = num_drugs
        data["protein"].num_nodes = num_proteins
        data["disease"].num_nodes = num_diseases
        data["sideeffect"].num_nodes = num_sideeffects

        # Collect all edges for degree computation for each node type
        # For degree features, we need ALL edges (incoming + outgoing) for each node
        # For bipartite edges (drug→protein), src is in drug space, dst in protein space!
        all_drug_edges = drug_drug.copy()
        all_drug_edges.extend(drug_drug)  # drug → drug (both directions)

        # For bipartite edges to other node types, only count outgoing from drug's perspective
        all_protein_edges = protein_protein.copy()
        all_protein_edges.extend(protein_protein)  # protein → protein (both directions)

        all_disease_edges = drug_disease.copy()  # disease nodes in disease space
        all_disease_edges.extend(protein_disease)  # disease nodes in disease space

        all_se_edges = drug_se.copy()  # sideeffect nodes in sideeffect space

        # Set node features based on node_feature_type parameter
        if self.node_feature_type == "degree":
            # Degree-based features (better for learning but slower to compute)
            data["drug"].x = (
                self._compute_degree_features(num_drugs, drug_drug)
                if drug_drug
                else torch.zeros((num_drugs, 2), dtype=torch.float)
            )
            data["protein"].x = (
                self._compute_degree_features(num_proteins, protein_protein)
                if protein_protein
                else torch.zeros((num_proteins, 2), dtype=torch.float)
            )
            data["disease"].x = (
                self._compute_degree_features(num_diseases, all_disease_edges)
                if all_disease_edges
                else torch.zeros((num_diseases, 2), dtype=torch.float)
            )
            data["sideeffect"].x = (
                self._compute_degree_features(num_sideeffects, all_se_edges)
                if all_se_edges
                else torch.zeros((num_sideeffects, 2), dtype=torch.float)
            )
        else:  # constant
            # Constant features (faster but may limit learning)
            data["drug"].x = torch.ones((num_drugs, 1), dtype=torch.float)
            data["protein"].x = torch.ones((num_proteins, 1), dtype=torch.float)
            data["disease"].x = torch.ones((num_diseases, 1), dtype=torch.float)
            data["sideeffect"].x = torch.ones((num_sideeffects, 1), dtype=torch.float)

        # Add edges to HeteroData
        if drug_drug:
            src, dst = zip(*drug_drug, strict=False)
            data["drug", "interacts", "drug"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        if drug_protein:
            src, dst = zip(*drug_protein, strict=False)
            data["drug", "targets", "protein"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        if drug_disease:
            src, dst = zip(*drug_disease, strict=False)
            data["drug", "treats", "disease"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        if drug_se:
            src, dst = zip(*drug_se, strict=False)
            data["drug", "causes", "sideeffect"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        if protein_protein:
            src, dst = zip(*protein_protein, strict=False)
            data["protein", "interacts", "protein"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        if protein_disease:
            src, dst = zip(*protein_disease, strict=False)
            data["protein", "associated_with", "disease"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        return data

    def _process_deepdr_net(self):
        """Process deepDR-Net dataset."""
        data = HeteroData()

        # Load edge lists
        drug_disease = self._load_edges(
            os.path.join(self.raw_dir, "drugDisease.txt"),
            subsample_ratio=self.subsample_ratio,
        )
        drug_drug = self._load_edges(
            os.path.join(self.raw_dir, "drugdrug.txt"),
            subsample_ratio=self.subsample_ratio,
        )
        drug_protein = self._load_edges(
            os.path.join(self.raw_dir, "drugProtein.txt"),
            subsample_ratio=self.subsample_ratio,
        )
        drug_se = self._load_edges(
            os.path.join(self.raw_dir, "drugsideEffect.txt"),
            subsample_ratio=self.subsample_ratio,
        )

        # Get max node IDs to determine node counts
        all_drug_ids = set()
        all_protein_ids = set()
        all_disease_ids = set()
        all_se_ids = set()

        for src, dst in drug_disease:
            all_drug_ids.add(src)
            all_disease_ids.add(dst)

        for src, dst in drug_drug:
            all_drug_ids.add(src)
            all_drug_ids.add(dst)

        for src, dst in drug_protein:
            all_drug_ids.add(src)
            all_protein_ids.add(dst)

        for src, dst in drug_se:
            all_drug_ids.add(src)
            all_se_ids.add(dst)

        # Set number of nodes and add constant features
        num_drugs = max(all_drug_ids) + 1 if all_drug_ids else 0
        num_proteins = max(all_protein_ids) + 1 if all_protein_ids else 0
        num_diseases = max(all_disease_ids) + 1 if all_disease_ids else 0
        num_sideeffects = max(all_se_ids) + 1 if all_se_ids else 0

        data["drug"].num_nodes = num_drugs
        data["protein"].num_nodes = num_proteins
        data["disease"].num_nodes = num_diseases
        data["sideeffect"].num_nodes = num_sideeffects

        # Collect all edges by node type for degree computation
        all_drug_edges = []
        all_protein_edges = []
        all_disease_edges = []
        all_se_edges = []

        if drug_drug:
            all_drug_edges.extend(drug_drug)
            all_drug_edges.extend([(d, s) for s, d in drug_drug])  # undirected
        if drug_protein:
            all_drug_edges.extend(drug_protein)
            all_protein_edges.extend([(d, s) for s, d in drug_protein])
        if drug_disease:
            all_drug_edges.extend(drug_disease)
            all_disease_edges.extend([(d, s) for s, d in drug_disease])
        if drug_se:
            all_drug_edges.extend(drug_se)
            all_se_edges.extend([(d, s) for s, d in drug_se])

        # Create node features based on node_feature_type parameter
        if self.node_feature_type == "degree":
            # Degree-based features: [log(in_degree+1), log(out_degree+1)]
            if num_drugs > 0:
                data["drug"].x = (
                    self._compute_degree_features(num_drugs, all_drug_edges)
                    if all_drug_edges
                    else torch.zeros((num_drugs, 2), dtype=torch.float)
                )
            if num_proteins > 0:
                data["protein"].x = (
                    self._compute_degree_features(num_proteins, all_protein_edges)
                    if all_protein_edges
                    else torch.zeros((num_proteins, 2), dtype=torch.float)
                )
            if num_diseases > 0:
                data["disease"].x = (
                    self._compute_degree_features(num_diseases, all_disease_edges)
                    if all_disease_edges
                    else torch.zeros((num_diseases, 2), dtype=torch.float)
                )
            if num_sideeffects > 0:
                data["sideeffect"].x = (
                    self._compute_degree_features(num_sideeffects, all_se_edges)
                    if all_se_edges
                    else torch.zeros((num_sideeffects, 2), dtype=torch.float)
                )
        else:
            # Constant features: all ones
            if num_drugs > 0:
                data["drug"].x = torch.ones((num_drugs, 1), dtype=torch.float)
            if num_proteins > 0:
                data["protein"].x = torch.ones((num_proteins, 1), dtype=torch.float)
            if num_diseases > 0:
                data["disease"].x = torch.ones((num_diseases, 1), dtype=torch.float)
            if num_sideeffects > 0:
                data["sideeffect"].x = torch.ones(
                    (num_sideeffects, 1), dtype=torch.float
                )

        # Add edges
        if drug_drug:
            src, dst = zip(*drug_drug, strict=False)
            data["drug", "interacts", "drug"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        if drug_protein:
            src, dst = zip(*drug_protein, strict=False)
            data["drug", "targets", "protein"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        if drug_disease:
            src, dst = zip(*drug_disease, strict=False)
            data["drug", "treats", "disease"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        if drug_se:
            src, dst = zip(*drug_se, strict=False)
            data["drug", "causes", "sideeffect"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )

        return data

    def _process_dda_network(self):
        """Process CTD-DDA or NDFRT-DDA dataset."""
        data = HeteroData()
        node_list_file = os.path.join(self.raw_dir, "node_list.txt")
        edge_file = os.path.join(self.raw_dir, f"{self.name}.edgelist")

        # Parse node list to get node types
        # CTD_DDA: 4 columns (index, CTD_id, UMLS_CUI, type)
        # NDFRT_DDA: 3 columns (index, UMLS_CUI, type)
        node_to_type = {}
        with open(node_list_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    idx = int(parts[0])
                    # Type is in last column (index -1)
                    node_type = parts[-1].strip()
                    node_to_type[idx] = node_type

        # Parse edges
        edges = []
        with open(edge_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        src, dst = int(parts[0]), int(parts[1])
                        edges.append((src, dst))
                    except ValueError:
                        continue

        # Separate nodes by type (accept 'chemical' or 'drug' as drug type)
        drugs = set()
        diseases = set()
        for src, dst in edges:
            if src in node_to_type and dst in node_to_type:
                src_type = node_to_type[src]
                dst_type = node_to_type[dst]
                if src_type in ["chemical", "drug"]:
                    drugs.add(src)
                elif src_type == "disease":
                    diseases.add(src)
                if dst_type in ["chemical", "drug"]:
                    drugs.add(dst)
                elif dst_type == "disease":
                    diseases.add(dst)

        # Create mapping to consecutive IDs
        unique_drugs = sorted(drugs)
        unique_diseases = sorted(diseases)

        drug_map = {old_id: new_id for new_id, old_id in enumerate(unique_drugs)}
        disease_map = {old_id: new_id for new_id, old_id in enumerate(unique_diseases)}

        # Remap edges (only keep edges between chemicals and diseases)
        src_list = []
        dst_list = []
        for src, dst in edges:
            if src in drug_map and dst in disease_map:
                src_list.append(drug_map[src])
                dst_list.append(disease_map[dst])

        # Set number of nodes and add constant features
        data["drug"].num_nodes = len(unique_drugs)
        data["disease"].num_nodes = len(unique_diseases)
        data["drug"].x = torch.ones((len(unique_drugs), 1), dtype=torch.float)
        data["disease"].x = torch.ones((len(unique_diseases), 1), dtype=torch.float)

        # Add edges
        data["drug", "treats", "disease"].edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )

        return data

    def _process_ddi_network(self):
        """Process DrugBank-DDI dataset."""
        data = HeteroData()
        node_list_file = os.path.join(self.raw_dir, "node_list.txt")
        edge_file = os.path.join(self.raw_dir, "DrugBank_DDI.edgelist")

        # Parse node list to get all drug nodes
        all_nodes = set()
        with open(node_list_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 1:
                    idx = int(parts[0])
                    all_nodes.add(idx)

        # Parse edges
        edges = []
        with open(edge_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        src, dst = int(parts[0]), int(parts[1])
                        edges.append((src, dst))
                    except ValueError:
                        continue

        # Get unique drugs from edges
        drugs = set()
        for src, dst in edges:
            drugs.add(src)
            drugs.add(dst)

        # Create mapping to consecutive IDs
        unique_drugs = sorted(drugs)
        drug_map = {old_id: new_id for new_id, old_id in enumerate(unique_drugs)}

        # Remap edges
        src_list = [drug_map[src] for src, dst in edges]
        dst_list = [drug_map[dst] for src, dst in edges]

        # Set number of nodes and create degree-based features
        data["drug"].num_nodes = len(unique_drugs)

        # Map edges to new IDs for degree computation
        remapped_edges = [(drug_map[src], drug_map[dst]) for src, dst in edges]

        # Create degree-based features (undirected graph)
        data["drug"].x = self._compute_degree_features(
            len(unique_drugs), remapped_edges
        )

        # Add edges
        data["drug", "interacts", "drug"].edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )

        return data

    def _process_ppi_network(self):
        """Process STRING-PPI dataset."""
        data = HeteroData()
        node_list_file = os.path.join(self.raw_dir, "node_list.txt")
        edge_file = os.path.join(self.raw_dir, "STRING_PPI.edgelist")

        # Parse node list to get all protein nodes
        all_nodes = set()
        with open(node_list_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 1:
                    idx = int(parts[0])
                    all_nodes.add(idx)

        # Parse edges
        edges = []
        with open(edge_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        src, dst = int(parts[0]), int(parts[1])
                        edges.append((src, dst))
                    except ValueError:
                        continue

        # Get unique proteins from edges
        proteins = set()
        for src, dst in edges:
            proteins.add(src)
            proteins.add(dst)

        # Create mapping to consecutive IDs
        unique_proteins = sorted(proteins)
        protein_map = {old_id: new_id for new_id, old_id in enumerate(unique_proteins)}

        # Remap edges
        src_list = [protein_map[src] for src, dst in edges]
        dst_list = [protein_map[dst] for src, dst in edges]

        # Set number of nodes and create degree-based features
        data["protein"].num_nodes = len(unique_proteins)

        # Map edges to new IDs for degree computation
        remapped_edges = [(protein_map[src], protein_map[dst]) for src, dst in edges]

        # Create degree-based features (undirected graph)
        data["protein"].x = self._compute_degree_features(
            len(unique_proteins), remapped_edges
        )

        # Add edges
        data["protein", "interacts", "protein"].edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )

        return data


class CTD_DDADataModule(LinkPredBase):
    """CTD Drug-Disease Association dataset for link prediction."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        node_feature_type: str = "degree",
    ):
        # Add suffix to cache path for different feature types
        cache_suffix = f"_{node_feature_type}" if node_feature_type != "degree" else ""
        self.biomedical_dataset = BiomedicalDataset(
            name="CTD_DDA",
            root=f"{data_dir}/ctd_dda{cache_suffix}",
            node_feature_type=node_feature_type,
        )
        super(CTD_DDADataModule, self).__init__(
            target=("drug", "treats", "disease"),
            rev_target=("disease", "rev_treats", "drug"),
            data_dir=data_dir,
            is_homogeneous=homogeneous,
            num_classes=1,
        )

    def download_data(self) -> HeteroData:
        """Return the processed HeteroData."""
        return self.biomedical_dataset[0]

    def __repr__(self):
        return "CTD_DDA"


class NDFRT_DDADataModule(LinkPredBase):
    """NDFRT Drug-Disease Association dataset for link prediction."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        node_feature_type: str = "degree",
    ):
        cache_suffix = f"_{node_feature_type}" if node_feature_type != "degree" else ""
        self.biomedical_dataset = BiomedicalDataset(
            name="NDFRT_DDA",
            root=f"{data_dir}/ndfrt_dda{cache_suffix}",
            node_feature_type=node_feature_type,
        )
        super(NDFRT_DDADataModule, self).__init__(
            target=("drug", "treats", "disease"),
            rev_target=("disease", "rev_treats", "drug"),
            data_dir=data_dir,
            is_homogeneous=homogeneous,
            num_classes=1,
        )

    def download_data(self) -> HeteroData:
        """Return the processed HeteroData."""
        return self.biomedical_dataset[0]

    def __repr__(self):
        return "NDFRT_DDA"


class DrugBankDDIDataModule(LinkPredBase):
    """DrugBank Drug-Drug Interaction dataset for link prediction."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        node_feature_type: str = "degree",
    ):
        cache_suffix = f"_{node_feature_type}" if node_feature_type != "degree" else ""
        self.biomedical_dataset = BiomedicalDataset(
            name="DrugBank_DDI",
            root=f"{data_dir}/drugbank_ddi{cache_suffix}",
            node_feature_type=node_feature_type,
        )
        super(DrugBankDDIDataModule, self).__init__(
            target=("drug", "interacts", "drug"),
            rev_target=None,
            data_dir=data_dir,
            is_homogeneous=homogeneous,
            num_classes=1,
        )

    def download_data(self) -> HeteroData:
        """Return the processed HeteroData."""
        return self.biomedical_dataset[0]

    def __repr__(self):
        return "DrugBank_DDI"


class STRINGPPIDataModule(LinkPredBase):
    """STRING Protein-Protein Interaction dataset for link prediction."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        node_feature_type: str = "degree",
    ):
        cache_suffix = f"_{node_feature_type}" if node_feature_type != "degree" else ""
        self.biomedical_dataset = BiomedicalDataset(
            name="STRING_PPI",
            root=f"{data_dir}/string_ppi{cache_suffix}",
            node_feature_type=node_feature_type,
        )
        super(STRINGPPIDataModule, self).__init__(
            target=("protein", "interacts", "protein"),
            rev_target=None,
            data_dir=data_dir,
            is_homogeneous=homogeneous,
            num_classes=1,
        )

    def download_data(self) -> HeteroData:
        """Return the processed HeteroData."""
        return self.biomedical_dataset[0]

    def __repr__(self):
        return "STRING_PPI"


class NeoDTINetDataModule(LinkPredBase):
    """NeoDTI complex biomedical network for Drug-Target Interaction prediction."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        subsample_ratio: float = 0.1,
        node_feature_type: str = "constant",
    ):
        # Add suffix for non-default feature types (default is 'constant' for NeoDTI)
        cache_suffix = (
            f"_{node_feature_type}" if node_feature_type != "constant" else ""
        )
        self.biomedical_dataset = BiomedicalDataset(
            name="NeoDTI-Net",
            root=f"{data_dir}/neodti_net{cache_suffix}",
            subsample_ratio=subsample_ratio,
            node_feature_type=node_feature_type,
        )
        super(NeoDTINetDataModule, self).__init__(
            target=("drug", "targets", "protein"),
            rev_target=("protein", "rev_targets", "drug"),
            data_dir=data_dir,
            is_homogeneous=homogeneous,
            num_classes=1,
        )

    def download_data(self) -> HeteroData:
        """Return the processed HeteroData."""
        return self.biomedical_dataset[0]

    def __repr__(self):
        return "NeoDTI_Net"


class DeepDRNetDataModule(LinkPredBase):
    """deepDR complex biomedical network for Drug-Disease Association prediction."""

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        homogeneous: bool = False,
        node_feature_type: str = "degree",
    ):
        cache_suffix = f"_{node_feature_type}" if node_feature_type != "degree" else ""
        self.biomedical_dataset = BiomedicalDataset(
            name="deepDR-Net",
            root=f"{data_dir}/deepdr_net{cache_suffix}",
            node_feature_type=node_feature_type,
        )
        super(DeepDRNetDataModule, self).__init__(
            target=("drug", "treats", "disease"),
            rev_target=("disease", "rev_treats", "drug"),
            data_dir=data_dir,
            is_homogeneous=homogeneous,
            num_classes=1,
        )

    def download_data(self) -> HeteroData:
        """Return the processed HeteroData."""
        return self.biomedical_dataset[0]

    def __repr__(self):
        return "DeepDR_Net"
