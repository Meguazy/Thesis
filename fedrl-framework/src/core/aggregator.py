#!/usr/bin/env python3
"""
FedRL Framework - Model Aggregation Engine

Implements various aggregation strategies for clustered federated learning,
including intra-cluster and inter-cluster aggregation methods.

Author: Based on Francesco Finucci's FedRL Framework
"""

import torch
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import copy
from datetime import datetime


class AggregationType(Enum):
    """Types of aggregation in clustered federated learning."""
    INTRA_CLUSTER = "intra_cluster"      # Within same chess style cluster
    INTER_CLUSTER = "inter_cluster"      # Across different chess style clusters
    GLOBAL = "global"                    # All nodes together (baseline)


class AggregationMethod(Enum):
    """Aggregation methods available."""
    FEDAVG = "fedavg"                           # Standard FedAvg
    WEIGHTED_FEDAVG = "weighted_fedavg"         # Performance-weighted FedAvg
    DIVERSITY_PRESERVING = "diversity_preserving"  # Novel diversity-preserving method
    SELECTIVE_SHARING = "selective_sharing"      # Selective parameter sharing


@dataclass
class NodeContribution:
    """Represents a node's contribution to aggregation."""
    node_id: str
    cluster_id: str
    model_parameters: Dict[str, torch.Tensor]
    training_samples: int
    performance_metrics: Dict[str, float]
    model_version: int
    contribution_weight: float = 1.0


@dataclass
class AggregationResult:
    """Result of an aggregation operation."""
    aggregated_parameters: Dict[str, torch.Tensor]
    aggregation_method: str
    participating_nodes: List[str]
    total_samples: int
    aggregation_weights: Dict[str, float]
    performance_metrics: Dict[str, Any]
    success: bool = True
    error_message: str = ""


class BaseAggregator(ABC):
    """Abstract base class for all aggregation methods."""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.logger = logging.getLogger(f"Aggregator-{method_name}")
    
    @abstractmethod
    def aggregate(self, contributions: List[NodeContribution]) -> AggregationResult:
        """Perform aggregation on node contributions."""
        pass
    
    def validate_contributions(self, contributions: List[NodeContribution]) -> bool:
        """Validate that contributions are compatible for aggregation."""
        if not contributions:
            self.logger.error("No contributions provided for aggregation")
            return False
        
        # Check that all contributions have the same parameter structure
        reference_keys = set(contributions[0].model_parameters.keys())
        
        for contrib in contributions[1:]:
            if set(contrib.model_parameters.keys()) != reference_keys:
                self.logger.error(f"Parameter structure mismatch in contribution from {contrib.node_id}")
                return False
            
            # Check parameter shapes
            for key in reference_keys:
                ref_shape = contributions[0].model_parameters[key].shape
                if contrib.model_parameters[key].shape != ref_shape:
                    self.logger.error(f"Parameter shape mismatch for {key} in {contrib.node_id}")
                    return False
        
        return True
    
    def calculate_sample_weights(self, contributions: List[NodeContribution]) -> Dict[str, float]:
        """Calculate weights based on training samples."""
        total_samples = sum(contrib.training_samples for contrib in contributions)
        
        if total_samples == 0:
            # Equal weights if no sample information
            weight = 1.0 / len(contributions)
            return {contrib.node_id: weight for contrib in contributions}
        
        return {
            contrib.node_id: contrib.training_samples / total_samples
            for contrib in contributions
        }


class FedAvgAggregator(BaseAggregator):
    """Standard Federated Averaging aggregation."""
    
    def __init__(self):
        super().__init__("FedAvg")
    
    def aggregate(self, contributions: List[NodeContribution]) -> AggregationResult:
        """Perform standard FedAvg aggregation."""
        if not self.validate_contributions(contributions):
            return AggregationResult(
                aggregated_parameters={},
                aggregation_method=self.method_name,
                participating_nodes=[],
                total_samples=0,
                aggregation_weights={},
                performance_metrics={},
                success=False,
                error_message="Contribution validation failed"
            )
        
        # Calculate weights based on training samples
        weights = self.calculate_sample_weights(contributions)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        reference_params = contributions[0].model_parameters
        
        for param_name in reference_params.keys():
            # Weighted average of parameters
            weighted_sum = torch.zeros_like(reference_params[param_name])
            
            for contrib in contributions:
                weight = weights[contrib.node_id]
                weighted_sum += weight * contrib.model_parameters[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        # Calculate performance metrics
        total_samples = sum(contrib.training_samples for contrib in contributions)
        avg_performance = {}
        
        if contributions[0].performance_metrics:
            for metric_name in contributions[0].performance_metrics.keys():
                weighted_metric = sum(
                    weights[contrib.node_id] * contrib.performance_metrics.get(metric_name, 0)
                    for contrib in contributions
                )
                avg_performance[metric_name] = weighted_metric
        
        self.logger.info(f"FedAvg aggregation complete: {len(contributions)} nodes, {total_samples} total samples")
        
        return AggregationResult(
            aggregated_parameters=aggregated_params,
            aggregation_method=self.method_name,
            participating_nodes=[contrib.node_id for contrib in contributions],
            total_samples=total_samples,
            aggregation_weights=weights,
            performance_metrics=avg_performance,
            success=True
        )


class WeightedFedAvgAggregator(BaseAggregator):
    """Performance-weighted Federated Averaging."""
    
    def __init__(self, performance_metric: str = "win_rate"):
        super().__init__("WeightedFedAvg")
        self.performance_metric = performance_metric
    
    def calculate_performance_weights(self, contributions: List[NodeContribution]) -> Dict[str, float]:
        """Calculate weights based on performance metrics."""
        performance_scores = []
        
        for contrib in contributions:
            score = contrib.performance_metrics.get(self.performance_metric, 0.0)
            # Ensure non-negative weights
            performance_scores.append(max(score, 0.1))
        
        total_performance = sum(performance_scores)
        
        if total_performance == 0:
            # Fall back to equal weights
            weight = 1.0 / len(contributions)
            return {contrib.node_id: weight for contrib in contributions}
        
        return {
            contrib.node_id: score / total_performance
            for contrib, score in zip(contributions, performance_scores)
        }
    
    def aggregate(self, contributions: List[NodeContribution]) -> AggregationResult:
        """Perform performance-weighted aggregation."""
        if not self.validate_contributions(contributions):
            return AggregationResult(
                aggregated_parameters={},
                aggregation_method=self.method_name,
                participating_nodes=[],
                total_samples=0,
                aggregation_weights={},
                performance_metrics={},
                success=False,
                error_message="Contribution validation failed"
            )
        
        # Calculate weights based on performance
        weights = self.calculate_performance_weights(contributions)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        reference_params = contributions[0].model_parameters
        
        for param_name in reference_params.keys():
            # Performance-weighted average of parameters
            weighted_sum = torch.zeros_like(reference_params[param_name])
            
            for contrib in contributions:
                weight = weights[contrib.node_id]
                weighted_sum += weight * contrib.model_parameters[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        total_samples = sum(contrib.training_samples for contrib in contributions)
        
        self.logger.info(f"Weighted FedAvg aggregation complete: {len(contributions)} nodes")
        
        return AggregationResult(
            aggregated_parameters=aggregated_params,
            aggregation_method=self.method_name,
            participating_nodes=[contrib.node_id for contrib in contributions],
            total_samples=total_samples,
            aggregation_weights=weights,
            performance_metrics={"weighting_metric": self.performance_metric},
            success=True
        )


class DiversityPreservingAggregator(BaseAggregator):
    """Novel diversity-preserving aggregation for maintaining chess playing styles."""
    
    def __init__(self, diversity_coefficient: float = 0.3):
        super().__init__("DiversityPreserving")
        self.diversity_coefficient = diversity_coefficient  # How much to preserve individual differences
    
    def identify_universal_parameters(self, contributions: List[NodeContribution]) -> List[str]:
        """Identify parameters that should be shared universally (e.g., tactical patterns).
        
        This is a simplified heuristic. In practice, this could be based on:
        - Layer analysis (early layers more universal, later layers more style-specific)
        - Parameter variance analysis across clusters
        - Domain knowledge about chess neural networks
        """
        if not contributions:
            return []
        
        reference_params = contributions[0].model_parameters
        universal_params = []
        
        for param_name in reference_params.keys():
            # Heuristic: consider some layers as universal
            if any(keyword in param_name.lower() for keyword in 
                   ['embedding', 'early', 'input', 'conv1', 'conv2']):
                universal_params.append(param_name)
        
        return universal_params
    
    def calculate_diversity_weights(self, contributions: List[NodeContribution]) -> Dict[str, float]:
        """Calculate weights that preserve diversity within the cluster."""
        # For intra-cluster aggregation, we still want some diversity preservation
        sample_weights = self.calculate_sample_weights(contributions)
        
        # Adjust weights to preserve some individual characteristics
        diversity_weights = {}
        
        for contrib in contributions:
            base_weight = sample_weights[contrib.node_id]
            # Reduce dominance of any single node to preserve diversity
            adjusted_weight = base_weight * (1 - self.diversity_coefficient) + \
                            (self.diversity_coefficient / len(contributions))
            diversity_weights[contrib.node_id] = adjusted_weight
        
        # Normalize weights
        total_weight = sum(diversity_weights.values())
        for node_id in diversity_weights:
            diversity_weights[node_id] /= total_weight
        
        return diversity_weights
    
    def aggregate(self, contributions: List[NodeContribution]) -> AggregationResult:
        """Perform diversity-preserving aggregation."""
        if not self.validate_contributions(contributions):
            return AggregationResult(
                aggregated_parameters={},
                aggregation_method=self.method_name,
                participating_nodes=[],
                total_samples=0,
                aggregation_weights={},
                performance_metrics={},
                success=False,
                error_message="Contribution validation failed"
            )
        
        # Identify universal vs style-specific parameters
        universal_params = self.identify_universal_parameters(contributions)
        
        # Calculate diversity-preserving weights
        weights = self.calculate_diversity_weights(contributions)
        
        # Aggregate parameters
        aggregated_params = {}
        reference_params = contributions[0].model_parameters
        
        for param_name in reference_params.keys():
            if param_name in universal_params:
                # Full aggregation for universal parameters
                weighted_sum = torch.zeros_like(reference_params[param_name])
                for contrib in contributions:
                    weight = weights[contrib.node_id]
                    weighted_sum += weight * contrib.model_parameters[param_name]
                aggregated_params[param_name] = weighted_sum
                
            else:
                # Partial aggregation for style-specific parameters
                weighted_sum = torch.zeros_like(reference_params[param_name])
                individual_sum = torch.zeros_like(reference_params[param_name])
                
                for contrib in contributions:
                    weight = weights[contrib.node_id]
                    param_value = contrib.model_parameters[param_name]
                    
                    # Weighted aggregation component
                    weighted_sum += weight * param_value
                    # Individual component (average of individual values)
                    individual_sum += param_value / len(contributions)
                
                # Combine aggregated and individual components
                aggregated_params[param_name] = \
                    (1 - self.diversity_coefficient) * weighted_sum + \
                    self.diversity_coefficient * individual_sum
        
        total_samples = sum(contrib.training_samples for contrib in contributions)
        
        self.logger.info(f"Diversity-preserving aggregation complete: {len(contributions)} nodes, "
                        f"universal params: {len(universal_params)}")
        
        return AggregationResult(
            aggregated_parameters=aggregated_params,
            aggregation_method=self.method_name,
            participating_nodes=[contrib.node_id for contrib in contributions],
            total_samples=total_samples,
            aggregation_weights=weights,
            performance_metrics={
                "diversity_coefficient": self.diversity_coefficient,
                "universal_params_count": len(universal_params),
                "style_specific_params_count": len(reference_params) - len(universal_params)
            },
            success=True
        )


class SelectiveSharingAggregator(BaseAggregator):
    """Selective knowledge transfer between different chess style clusters."""
    
    def __init__(self, sharing_threshold: float = 0.8):
        super().__init__("SelectiveSharing")
        self.sharing_threshold = sharing_threshold  # Minimum consensus for parameter sharing
    
    def calculate_parameter_consensus(self, contributions: List[NodeContribution]) -> Dict[str, float]:
        """Calculate consensus score for each parameter across clusters."""
        if len(contributions) < 2:
            return {}
        
        consensus_scores = {}
        reference_params = contributions[0].model_parameters
        
        for param_name in reference_params.keys():
            # Calculate pairwise similarities
            similarities = []
            
            for i in range(len(contributions)):
                for j in range(i + 1, len(contributions)):
                    param_i = contributions[i].model_parameters[param_name]
                    param_j = contributions[j].model_parameters[param_name]
                    
                    # Cosine similarity between parameter tensors
                    param_i_flat = param_i.flatten()
                    param_j_flat = param_j.flatten()
                    
                    dot_product = torch.dot(param_i_flat, param_j_flat)
                    norm_i = torch.norm(param_i_flat)
                    norm_j = torch.norm(param_j_flat)
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = dot_product / (norm_i * norm_j)
                        similarities.append(float(similarity))
            
            # Average similarity as consensus score
            consensus_scores[param_name] = np.mean(similarities) if similarities else 0.0
        
        return consensus_scores
    
    def aggregate(self, contributions: List[NodeContribution]) -> AggregationResult:
        """Perform selective sharing aggregation between clusters."""
        if not self.validate_contributions(contributions):
            return AggregationResult(
                aggregated_parameters={},
                aggregation_method=self.method_name,
                participating_nodes=[],
                total_samples=0,
                aggregation_weights={},
                performance_metrics={},
                success=False,
                error_message="Contribution validation failed"
            )
        
        # Calculate parameter consensus across clusters
        consensus_scores = self.calculate_parameter_consensus(contributions)
        
        # Standard sample-based weights
        weights = self.calculate_sample_weights(contributions)
        
        # Aggregate parameters selectively
        aggregated_params = {}
        shared_params = []
        preserved_params = []
        
        reference_params = contributions[0].model_parameters
        
        for param_name in reference_params.keys():
            consensus = consensus_scores.get(param_name, 0.0)
            
            if consensus >= self.sharing_threshold:
                # High consensus: aggregate this parameter
                weighted_sum = torch.zeros_like(reference_params[param_name])
                
                for contrib in contributions:
                    weight = weights[contrib.node_id]
                    weighted_sum += weight * contrib.model_parameters[param_name]
                
                aggregated_params[param_name] = weighted_sum
                shared_params.append(param_name)
                
            else:
                # Low consensus: preserve cluster-specific values
                # Take the parameter from the highest-performing node
                best_contrib = max(contributions, 
                                 key=lambda c: c.performance_metrics.get('win_rate', 0.0))
                aggregated_params[param_name] = best_contrib.model_parameters[param_name].clone()
                preserved_params.append(param_name)
        
        total_samples = sum(contrib.training_samples for contrib in contributions)
        
        self.logger.info(f"Selective sharing aggregation complete: {len(shared_params)} shared, "
                        f"{len(preserved_params)} preserved parameters")
        
        return AggregationResult(
            aggregated_parameters=aggregated_params,
            aggregation_method=self.method_name,
            participating_nodes=[contrib.node_id for contrib in contributions],
            total_samples=total_samples,
            aggregation_weights=weights,
            performance_metrics={
                "sharing_threshold": self.sharing_threshold,
                "shared_params_count": len(shared_params),
                "preserved_params_count": len(preserved_params),
                "avg_consensus_score": np.mean(list(consensus_scores.values())) if consensus_scores else 0.0,
                "shared_parameters": shared_params,
                "preserved_parameters": preserved_params
            },
            success=True
        )


class ClusterAwareAggregator(BaseAggregator):
    """Aggregator that adapts strategy based on cluster composition and training phase."""
    
    def __init__(self, phase_strategies: Optional[Dict[str, str]] = None):
        super().__init__("ClusterAware")
        self.phase_strategies = phase_strategies or {
            "individual_development": "fedavg",
            "intra_cluster": "diversity_preserving", 
            "inter_cluster": "selective_sharing"
        }
        
        # Initialize sub-aggregators
        self.sub_aggregators = {
            "fedavg": FedAvgAggregator(),
            "weighted_fedavg": WeightedFedAvgAggregator(),
            "diversity_preserving": DiversityPreservingAggregator(),
            "selective_sharing": SelectiveSharingAggregator()
        }
    
    def select_strategy(self, contributions: List[NodeContribution], 
                       training_phase: str) -> BaseAggregator:
        """Select appropriate aggregation strategy based on context."""
        
        # Use phase-specific strategy if available
        if training_phase in self.phase_strategies:
            strategy_name = self.phase_strategies[training_phase]
            if strategy_name in self.sub_aggregators:
                return self.sub_aggregators[strategy_name]
        
        # Fallback logic based on cluster composition
        cluster_ids = set(contrib.cluster_id for contrib in contributions)
        
        if len(cluster_ids) == 1:
            # Intra-cluster: use diversity-preserving
            return self.sub_aggregators["diversity_preserving"]
        else:
            # Inter-cluster: use selective sharing
            return self.sub_aggregators["selective_sharing"]
    
    def aggregate(self, contributions: List[NodeContribution], 
                 training_phase: str = "intra_cluster") -> AggregationResult:
        """Perform context-aware aggregation."""
        
        if not self.validate_contributions(contributions):
            return AggregationResult(
                aggregated_parameters={},
                aggregation_method=self.method_name,
                participating_nodes=[],
                total_samples=0,
                aggregation_weights={},
                performance_metrics={},
                success=False,
                error_message="Contribution validation failed"
            )
        
        # Select and use appropriate strategy
        selected_aggregator = self.select_strategy(contributions, training_phase)
        result = selected_aggregator.aggregate(contributions)
        
        # Update method name to include context
        result.aggregation_method = f"{self.method_name}({selected_aggregator.method_name})"
        result.performance_metrics["selected_strategy"] = selected_aggregator.method_name
        result.performance_metrics["training_phase"] = training_phase
        
        return result


class AggregationEngine:
    """Main aggregation engine that manages different aggregation strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize aggregation engine with configuration."""
        self.config = config or {}
        
        # Initialize aggregators
        self.aggregators = {
            AggregationMethod.FEDAVG: FedAvgAggregator(),
            AggregationMethod.WEIGHTED_FEDAVG: WeightedFedAvgAggregator(
                performance_metric=self.config.get('performance_metric', 'win_rate')
            ),
            AggregationMethod.DIVERSITY_PRESERVING: DiversityPreservingAggregator(
                diversity_coefficient=self.config.get('diversity_coefficient', 0.3)
            ),
            AggregationMethod.SELECTIVE_SHARING: SelectiveSharingAggregator(
                sharing_threshold=self.config.get('sharing_threshold', 0.8)
            )
        }
        
        # Add cluster-aware aggregator
        self.cluster_aware_aggregator = ClusterAwareAggregator(
            self.config.get('phase_strategies')
        )
        
        self.logger = logging.getLogger("AggregationEngine")
        
        # Statistics tracking
        self.aggregation_history = []
        self.cluster_performance_history = {}
    
    def aggregate_intra_cluster(self, contributions: List[NodeContribution], 
                               method: AggregationMethod = AggregationMethod.DIVERSITY_PRESERVING,
                               training_phase: str = "intra_cluster") -> AggregationResult:
        """Perform intra-cluster aggregation (within same chess style)."""
        self.logger.info(f"Starting intra-cluster aggregation: {len(contributions)} nodes, method: {method.value}")
        
        # Validate that all contributions are from the same cluster
        cluster_ids = set(contrib.cluster_id for contrib in contributions)
        if len(cluster_ids) > 1:
            error_msg = f"Mixed cluster contributions for intra-cluster aggregation: {cluster_ids}"
            self.logger.error(error_msg)
            return AggregationResult(
                aggregated_parameters={},
                aggregation_method=method.value,
                participating_nodes=[],
                total_samples=0,
                aggregation_weights={},
                performance_metrics={},
                success=False,
                error_message=error_msg
            )
        
        cluster_id = list(cluster_ids)[0] if cluster_ids else "unknown"
        
        try:
            # Perform aggregation
            aggregator = self.aggregators[method]
            result = aggregator.aggregate(contributions)
            result.aggregation_method = f"intra_cluster_{method.value}"
            
            # Track performance history
            self._update_cluster_performance(cluster_id, result)
            
            # Log success
            self.logger.info(f"Intra-cluster aggregation successful for {cluster_id}: "
                           f"{result.total_samples} samples, {len(result.participating_nodes)} nodes")
            
            return result
            
        except Exception as e:
            error_msg = f"Intra-cluster aggregation failed: {str(e)}"
            self.logger.error(error_msg)
            return AggregationResult(
                aggregated_parameters={},
                aggregation_method=f"intra_cluster_{method.value}",
                participating_nodes=[contrib.node_id for contrib in contributions],
                total_samples=0,
                aggregation_weights={},
                performance_metrics={},
                success=False,
                error_message=error_msg
            )
    
    def aggregate_inter_cluster(self, cluster_contributions: Dict[str, List[NodeContribution]],
                               method: AggregationMethod = AggregationMethod.SELECTIVE_SHARING,
                               training_phase: str = "inter_cluster") -> Dict[str, AggregationResult]:
        """Perform inter-cluster aggregation (across different chess styles)."""
        self.logger.info(f"Starting inter-cluster aggregation: {len(cluster_contributions)} clusters, method: {method.value}")
        
        results = {}
        
        try:
            if method == AggregationMethod.SELECTIVE_SHARING:
                # For selective sharing, analyze all clusters together
                all_contributions = []
                cluster_representatives = {}
                
                for cluster_id, contributions in cluster_contributions.items():
                    if contributions:
                        # Select best representative from each cluster
                        best_contrib = max(contributions, 
                                         key=lambda c: c.performance_metrics.get('win_rate', 0.0))
                        all_contributions.append(best_contrib)
                        cluster_representatives[cluster_id] = best_contrib
                
                if len(all_contributions) > 1:
                    # Perform selective sharing analysis
                    aggregator = self.aggregators[method]
                    shared_knowledge = aggregator.aggregate(all_contributions)
                    
                    # Apply shared knowledge to each cluster's nodes
                    for cluster_id, contributions in cluster_contributions.items():
                        if contributions:
                            # Create cluster-specific result
                            cluster_result = copy.deepcopy(shared_knowledge)
                            cluster_result.aggregation_method = f"inter_cluster_{method.value}"
                            cluster_result.participating_nodes = [c.node_id for c in contributions]
                            
                            # Adjust weights for cluster-specific application
                            cluster_weights = {}
                            total_samples = sum(c.training_samples for c in contributions)
                            for contrib in contributions:
                                cluster_weights[contrib.node_id] = contrib.training_samples / total_samples if total_samples > 0 else 1.0 / len(contributions)
                            
                            cluster_result.aggregation_weights = cluster_weights
                            cluster_result.total_samples = total_samples
                            
                            results[cluster_id] = cluster_result
                
                else:
                    self.logger.warning("Not enough clusters for inter-cluster selective sharing")
                    
            else:
                # For other methods, apply to each cluster independently
                for cluster_id, contributions in cluster_contributions.items():
                    if contributions:
                        aggregator = self.aggregators[method]
                        result = aggregator.aggregate(contributions)
                        result.aggregation_method = f"inter_cluster_{method.value}"
                        results[cluster_id] = result
            
            # Update performance tracking
            for cluster_id, result in results.items():
                self._update_cluster_performance(cluster_id, result)
            
            self.logger.info(f"Inter-cluster aggregation completed for {len(results)} clusters")
            return results
            
        except Exception as e:
            error_msg = f"Inter-cluster aggregation failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Return error results for all clusters
            error_results = {}
            for cluster_id in cluster_contributions.keys():
                error_results[cluster_id] = AggregationResult(
                    aggregated_parameters={},
                    aggregation_method=f"inter_cluster_{method.value}",
                    participating_nodes=[],
                    total_samples=0,
                    aggregation_weights={},
                    performance_metrics={},
                    success=False,
                    error_message=error_msg
                )
            return error_results
    
    def aggregate_adaptive(self, contributions: List[NodeContribution],
                          training_phase: str) -> AggregationResult:
        """Perform adaptive aggregation based on training phase and cluster composition."""
        return self.cluster_aware_aggregator.aggregate(contributions, training_phase)
    
    def _update_cluster_performance(self, cluster_id: str, result: AggregationResult):
        """Update performance tracking for a cluster."""
        if cluster_id not in self.cluster_performance_history:
            self.cluster_performance_history[cluster_id] = []
        
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'method': result.aggregation_method,
            'participating_nodes': len(result.participating_nodes),
            'total_samples': result.total_samples,
            'success': result.success,
            'metrics': result.performance_metrics
        }
        
        self.cluster_performance_history[cluster_id].append(performance_record)
        
        # Keep only recent history (last 100 records)
        if len(self.cluster_performance_history[cluster_id]) > 100:
            self.cluster_performance_history[cluster_id] = self.cluster_performance_history[cluster_id][-100:]
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about aggregation operations."""
        stats = {
            "available_methods": [method.value for method in AggregationMethod],
            "aggregator_count": len(self.aggregators),
            "config": self.config,
            "cluster_count": len(self.cluster_performance_history),
            "clusters": {}
        }
        
        # Add cluster-specific statistics
        for cluster_id, history in self.cluster_performance_history.items():
            if history:
                successful_aggregations = sum(1 for h in history if h['success'])
                total_aggregations = len(history)
                
                stats["clusters"][cluster_id] = {
                    "total_aggregations": total_aggregations,
                    "successful_aggregations": successful_aggregations,
                    "success_rate": successful_aggregations / total_aggregations if total_aggregations > 0 else 0.0,
                    "last_aggregation": history[-1]['timestamp'] if history else None,
                    "avg_participating_nodes": np.mean([h['participating_nodes'] for h in history]),
                    "total_samples_processed": sum(h['total_samples'] for h in history)
                }
        
        return stats
    
    def get_cluster_performance_trends(self, cluster_id: str) -> Dict[str, Any]:
        """Get performance trends for a specific cluster."""
        if cluster_id not in self.cluster_performance_history:
            return {}
        
        history = self.cluster_performance_history[cluster_id]
        if not history:
            return {}
        
        # Extract time series data
        timestamps = [h['timestamp'] for h in history]
        success_rates = []
        sample_counts = [h['total_samples'] for h in history]
        node_counts = [h['participating_nodes'] for h in history]
        
        # Calculate rolling success rate
        window_size = min(10, len(history))
        for i in range(len(history)):
            start_idx = max(0, i - window_size + 1)
            window = history[start_idx:i+1]
            window_success_rate = sum(1 for h in window if h['success']) / len(window)
            success_rates.append(window_success_rate)
        
        return {
            "cluster_id": cluster_id,
            "data_points": len(history),
            "time_range": {
                "start": timestamps[0],
                "end": timestamps[-1]
            },
            "trends": {
                "success_rates": success_rates,
                "sample_counts": sample_counts,
                "node_counts": node_counts,
                "timestamps": timestamps
            },
            "summary": {
                "avg_success_rate": np.mean(success_rates),
                "avg_samples": np.mean(sample_counts),
                "avg_nodes": np.mean(node_counts),
                "latest_success_rate": success_rates[-1] if success_rates else 0.0
            }
        }
    
    def export_aggregation_history(self, filepath: str):
        """Export aggregation history to file."""
        import json
        
        export_data = {
            "config": self.config,
            "cluster_performance_history": self.cluster_performance_history,
            "export_timestamp": datetime.now().isoformat(),
            "stats": self.get_aggregation_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Aggregation history exported to {filepath}")


# === UTILITY FUNCTIONS ===

def create_dummy_contribution(node_id: str, cluster_id: str, 
                            param_shapes: Dict[str, Tuple],
                            performance_override: Optional[Dict[str, float]] = None) -> NodeContribution:
    """Create a dummy contribution for testing purposes."""
    model_params = {}
    
    for param_name, shape in param_shapes.items():
        # Add some cluster-specific patterns to make aggregation more realistic
        if cluster_id == "tactical_cluster":
            model_params[param_name] = torch.randn(*shape) * 0.1 + 0.2  # Slight positive bias
        elif cluster_id == "positional_cluster":
            model_params[param_name] = torch.randn(*shape) * 0.1 - 0.1  # Slight negative bias
        else:  # dynamic
            model_params[param_name] = torch.randn(*shape) * 0.15  # More variance
    
    # Default performance metrics with some cluster-specific tendencies
    default_metrics = {
        "win_rate": np.random.uniform(0.4, 0.8),
        "training_loss": np.random.uniform(0.1, 0.5),
        "eco_adherence_rate": np.random.uniform(0.85, 1.0),
        "avg_game_length": np.random.uniform(30, 80)
    }
    
    if performance_override:
        default_metrics.update(performance_override)
    
    return NodeContribution(
        node_id=node_id,
        cluster_id=cluster_id,
        model_parameters=model_params,
        training_samples=np.random.randint(50, 200),
        performance_metrics=default_metrics,
        model_version=1
    )


def compare_aggregation_methods(contributions: List[NodeContribution], 
                              training_phase: str = "intra_cluster") -> Dict[str, AggregationResult]:
    """Compare different aggregation methods on the same contributions."""
    engine = AggregationEngine()
    results = {}
    
    methods = [
        AggregationMethod.FEDAVG,
        AggregationMethod.WEIGHTED_FEDAVG,
        AggregationMethod.DIVERSITY_PRESERVING
    ]
    
    for method in methods:
        result = engine.aggregate_intra_cluster(contributions, method, training_phase)
        results[method.value] = result
    
    # Also test adaptive aggregation
    adaptive_result = engine.aggregate_adaptive(contributions, training_phase)
    results["adaptive"] = adaptive_result
    
    return results


def analyze_parameter_diversity(contributions: List[NodeContribution]) -> Dict[str, float]:
    """Analyze diversity of parameters across contributions."""
    if len(contributions) < 2:
        return {}
    
    diversity_scores = {}
    reference_params = contributions[0].model_parameters
    
    for param_name in reference_params.keys():
        # Calculate pairwise distances
        distances = []
        
        for i in range(len(contributions)):
            for j in range(i + 1, len(contributions)):
                param_i = contributions[i].model_parameters[param_name].flatten()
                param_j = contributions[j].model_parameters[param_name].flatten()
                
                # L2 distance normalized by parameter magnitude
                distance = torch.norm(param_i - param_j).item()
                norm_factor = (torch.norm(param_i) + torch.norm(param_j)).item() / 2
                
                if norm_factor > 0:
                    normalized_distance = distance / norm_factor
                    distances.append(normalized_distance)
        
        # Average normalized distance as diversity score
        diversity_scores[param_name] = np.mean(distances) if distances else 0.0
    
    return diversity_scores


def simulate_federated_round(clusters: Dict[str, List[NodeContribution]], 
                           engine: AggregationEngine,
                           training_phase: str = "intra_cluster") -> Dict[str, Any]:
    """Simulate a complete federated learning round with multiple clusters."""
    round_results = {
        "phase": training_phase,
        "timestamp": datetime.now().isoformat(),
        "cluster_results": {},
        "summary": {}
    }
    
    total_nodes = 0
    total_samples = 0
    successful_clusters = 0
    
    if training_phase == "intra_cluster":
        # Perform intra-cluster aggregation for each cluster
        for cluster_id, contributions in clusters.items():
            if contributions:
                result = engine.aggregate_intra_cluster(
                    contributions, 
                    AggregationMethod.DIVERSITY_PRESERVING,
                    training_phase
                )
                round_results["cluster_results"][cluster_id] = {
                    "success": result.success,
                    "nodes": len(result.participating_nodes),
                    "samples": result.total_samples,
                    "method": result.aggregation_method,
                    "error": result.error_message if not result.success else None
                }
                
                total_nodes += len(result.participating_nodes)
                total_samples += result.total_samples
                if result.success:
                    successful_clusters += 1
    
    elif training_phase == "inter_cluster":
        # Perform inter-cluster aggregation
        inter_results = engine.aggregate_inter_cluster(
            clusters,
            AggregationMethod.SELECTIVE_SHARING,
            training_phase
        )
        
        for cluster_id, result in inter_results.items():
            round_results["cluster_results"][cluster_id] = {
                "success": result.success,
                "nodes": len(result.participating_nodes),
                "samples": result.total_samples,
                "method": result.aggregation_method,
                "shared_params": result.performance_metrics.get("shared_params_count", 0),
                "preserved_params": result.performance_metrics.get("preserved_params_count", 0),
                "error": result.error_message if not result.success else None
            }
            
            total_nodes += len(result.participating_nodes)
            total_samples += result.total_samples
            if result.success:
                successful_clusters += 1
    
    # Summary statistics
    round_results["summary"] = {
        "total_clusters": len(clusters),
        "successful_clusters": successful_clusters,
        "success_rate": successful_clusters / len(clusters) if clusters else 0.0,
        "total_nodes": total_nodes,
        "total_samples": total_samples,
        "avg_nodes_per_cluster": total_nodes / len(clusters) if clusters else 0.0
    }
    
    return round_results


# === EXAMPLE USAGE AND DEMONSTRATION ===

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("FedRL Aggregation Engine - Complete Implementation Demo")
    print("=" * 60)
    
    # Create realistic test scenario with chess-specific parameter shapes
    param_shapes = {
        "embedding.weight": (8192, 512),      # Chess position embeddings
        "conv1.weight": (64, 8, 3, 3),        # First convolutional layer
        "conv1.bias": (64,),
        "conv2.weight": (128, 64, 3, 3),      # Second convolutional layer
        "conv2.bias": (128,),
        "residual_block1.weight": (256, 128, 3, 3),  # Residual connections
        "residual_block1.bias": (256,),
        "value_head.weight": (1, 256),        # Value estimation head
        "value_head.bias": (1,),
        "policy_head.weight": (4096, 256),    # Policy head (chess moves)
        "policy_head.bias": (4096,)
    }
    
    # Create contributions for different clusters with realistic chess performance
    print("\n1. Creating Test Data")
    print("-" * 30)
    
    tactical_contributions = [
        create_dummy_contribution(f"tactical_node_{i}", "tactical_cluster", param_shapes, 
                                {"win_rate": np.random.uniform(0.6, 0.8), "avg_game_length": np.random.uniform(25, 45)})
        for i in range(4)
    ]
    
    positional_contributions = [
        create_dummy_contribution(f"positional_node_{i}", "positional_cluster", param_shapes,
                                {"win_rate": np.random.uniform(0.5, 0.7), "avg_game_length": np.random.uniform(45, 75)})
        for i in range(3)
    ]
    
    dynamic_contributions = [
        create_dummy_contribution(f"dynamic_node_{i}", "dynamic_cluster", param_shapes,
                                {"win_rate": np.random.uniform(0.55, 0.75), "avg_game_length": np.random.uniform(35, 65)})
        for i in range(3)
    ]
    
    print(f"Created tactical cluster: {len(tactical_contributions)} nodes")
    print(f"Created positional cluster: {len(positional_contributions)} nodes") 
    print(f"Created dynamic cluster: {len(dynamic_contributions)} nodes")
    
    # Initialize engine with comprehensive configuration
    print("\n2. Initializing Aggregation Engine")
    print("-" * 30)
    
    config = {
        "diversity_coefficient": 0.4,      # Higher diversity preservation
        "sharing_threshold": 0.75,         # Moderate consensus threshold
        "performance_metric": "win_rate",  # Weight by chess performance
        "phase_strategies": {
            "individual_development": "fedavg",
            "intra_cluster": "diversity_preserving", 
            "inter_cluster": "selective_sharing"
        }
    }
    
    engine = AggregationEngine(config)
    print(f"Engine initialized with config: {config}")
    
    # Test intra-cluster aggregation
    print("\n3. Testing Intra-Cluster Aggregation")
    print("-" * 30)
    
    tactical_result = engine.aggregate_intra_cluster(
        tactical_contributions, 
        AggregationMethod.DIVERSITY_PRESERVING,
        "intra_cluster"
    )
    
    print(f"Tactical Cluster Results:")
    print(f"  Success: {tactical_result.success}")
    print(f"  Method: {tactical_result.aggregation_method}")
    print(f"  Participating nodes: {len(tactical_result.participating_nodes)}")
    print(f"  Total samples: {tactical_result.total_samples}")
    print(f"  Universal params: {tactical_result.performance_metrics.get('universal_params_count', 'N/A')}")
    print(f"  Style-specific params: {tactical_result.performance_metrics.get('style_specific_params_count', 'N/A')}")
    
    # Test inter-cluster aggregation
    print("\n4. Testing Inter-Cluster Aggregation")
    print("-" * 30)
    
    all_clusters = {
        "tactical_cluster": tactical_contributions,
        "positional_cluster": positional_contributions,
        "dynamic_cluster": dynamic_contributions
    }
    
    inter_results = engine.aggregate_inter_cluster(
        all_clusters,
        AggregationMethod.SELECTIVE_SHARING,
        "inter_cluster"
    )
    
    print(f"Inter-Cluster Results:")
    for cluster_id, result in inter_results.items():
        print(f"  {cluster_id}:")
        print(f"    Success: {result.success}")
        print(f"    Method: {result.aggregation_method}")
        print(f"    Shared parameters: {result.performance_metrics.get('shared_params_count', 'N/A')}")
        print(f"    Preserved parameters: {result.performance_metrics.get('preserved_params_count', 'N/A')}")
        print(f"    Avg consensus: {result.performance_metrics.get('avg_consensus_score', 'N/A'):.3f}")
    
    # Test method comparison
    print("\n5. Comparing Aggregation Methods")
    print("-" * 30)
    
    method_comparison = compare_aggregation_methods(tactical_contributions, "intra_cluster")
    
    print("Method Comparison Results:")
    for method, result in method_comparison.items():
        print(f"  {method:<20}: Success={str(result.success):<5} | "
              f"Samples={result.total_samples:>4} | "
              f"Method={result.aggregation_method}")
    
    # Analyze parameter diversity
    print("\n6. Parameter Diversity Analysis")
    print("-" * 30)
    
    diversity_analysis = analyze_parameter_diversity(tactical_contributions)
    print("Parameter Diversity Scores (higher = more diverse):")
    
    sorted_diversity = sorted(diversity_analysis.items(), key=lambda x: x[1], reverse=True)
    for param_name, score in sorted_diversity[:8]:  # Show top 8 most diverse parameters
        print(f"  {param_name:<25}: {score:.4f}")
    
    if len(sorted_diversity) > 8:
        print(f"  ... and {len(sorted_diversity) - 8} more parameters")
    
    # Simulate complete federated rounds
    print("\n7. Simulating Complete Federated Rounds")
    print("-" * 30)
    
    # Intra-cluster round
    intra_round = simulate_federated_round(all_clusters, engine, "intra_cluster")
    print(f"Intra-Cluster Round:")
    print(f"  Success rate: {intra_round['summary']['success_rate']:.1%}")
    print(f"  Total nodes: {intra_round['summary']['total_nodes']}")
    print(f"  Total samples: {intra_round['summary']['total_samples']}")
    
    # Inter-cluster round  
    inter_round = simulate_federated_round(all_clusters, engine, "inter_cluster")
    print(f"Inter-Cluster Round:")
    print(f"  Success rate: {inter_round['summary']['success_rate']:.1%}")
    print(f"  Avg shared params per cluster: {np.mean([
        r['shared_params'] for r in inter_round['cluster_results'].values() 
        if 'shared_params' in r
    ]):.1f}")
    
    # Engine statistics
    print("\n8. Engine Performance Statistics")
    print("-" * 30)
    
    stats = engine.get_aggregation_stats()
    print(f"Available methods: {', '.join(stats['available_methods'])}")
    print(f"Clusters tracked: {stats['cluster_count']}")
    
    for cluster_id, cluster_stats in stats['clusters'].items():
        print(f"{cluster_id}:")
        print(f"  Total aggregations: {cluster_stats['total_aggregations']}")
        print(f"  Success rate: {cluster_stats['success_rate']:.1%}")
        print(f"  Avg participating nodes: {cluster_stats['avg_participating_nodes']:.1f}")
        print(f"  Total samples processed: {cluster_stats['total_samples_processed']}")
    
    # Export results
    print("\n9. Exporting Results")
    print("-" * 30)
    
    try:
        engine.export_aggregation_history("aggregation_history.json")
        print("✓ Aggregation history exported to 'aggregation_history.json'")
    except Exception as e:
        print(f"✗ Export failed: {e}")
    
    print("\n" + "=" * 60)
    print("FedRL Aggregation Engine Demo Complete!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("• Multiple aggregation strategies (FedAvg, Weighted, Diversity-Preserving, Selective)")
    print("• Intra-cluster and inter-cluster aggregation")
    print("• Chess-style specific parameter handling")
    print("• Performance tracking and analytics")
    print("• Parameter diversity analysis")
    print("• Complete federated round simulation")
    print("• Export capabilities for research analysis")