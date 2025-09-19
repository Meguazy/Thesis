"""
FedRL Framework - Central Server Implementation

This implements the central coordination server for clustered federated learning
with model serving capabilities.

Author: Based on Francesco Finucci's FedRL Framework
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta, timezone
import websockets
from websockets.server import WebSocketServerProtocol
from pathlib import Path
import torch
import threading
from concurrent.futures import ThreadPoolExecutor


class TrainingPhase(Enum):
    """Training phases in the clustered federated learning framework."""
    INDIVIDUAL_DEVELOPMENT = "individual_development"
    INTRA_CLUSTER = "intra_cluster"
    INTER_CLUSTER = "inter_cluster"
    COMPLETED = "completed"
    

class ChessStyle(Enum):
    """Chess styles for model training."""
    TACTICAL = "tactical"
    POSITIONAL = "positional"
    DYNAMIC = "dynamic"
    

class NodeStatus(Enum):
    """Status of a node in the federated learning framework."""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    TRAINING = "training"
    READY_FOR_AGGREGATION = "ready_for_aggregation"
    AGGREGATING = "aggregating"
    WAITING = "waiting"
    

class NodeInfo(Enum):
    """Information about a node in the federated learning framework."""
    node_id: str
    style: ChessStyle
    cluster_id: str
    status: NodeStatus
    websocket: Optional[Any] = None
    last_heartbeat: Optional[datetime] = None
    model_version: int = 0
    training_iterations: int = 0
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
            

class ClusterInfo(Enum):
    """Information about a cluster in the federated learning framework."""
    cluster_id: str
    style: ChessStyle
    eco_codes: List[str]
    nodes: Dict[str, NodeInfo]
    current_model_version: int = 1
    training_phase: TrainingPhase = TrainingPhase.INDIVIDUAL_DEVELOPMENT
    aggregation_round: int = 0
    last_aggregation: Optional[datetime] = None
    
    def __post_init__(self):
        if self.nodes is None:
            self.nodes = {}
            

class CentralModelRegistry:
    """Manages central models for each cluster"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Central models for each cluster
        self.models: Dict[ChessStyle, torch.nn.Module] = {}
        self.model_metadata: Dict[ChessStyle, Dict[str, Any]] = {}
        
        # Initialize model storage directories
        for style in ChessStyle:
            style_dir = self.model_dir / style.value
            style_dir.mkdir(exist_ok=True)
            (style_dir / "versions").mkdir(exist_ok=True)
            
    def save_model(self, style: ChessStyle, version: Optional[int] = None) -> Optional[torch.nn.Module]:
        """Saves the model for a given style and version."""
        style_dir = self.model_dir / style.value
        
        if version is None:
            model_path = style_dir / "current.pth"
        else:
            model_path = style_dir / "versions" / f"v{version}.pth"
            
        if not model_path.exists():
            logging.warning(f"Model path {model_path} does not exist.")
            return None
        
        try:
            # TODO: Replace with actual model class
            # For now, we'll assume a basic model structure
            # This should be replaced with actual model architecture loading
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_state = torch.load(model_path, map_location=device)
            logging.info(f"Loaded model for style {style.value} version {version if version else 'current'}.")
            return model_state
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")
            return None

    def get_model_metadata(self, style: ChessStyle) -> Optional[Dict[str, Any]]:
        """Returns metadata for the model of a given style."""
        if style in self.model_metadata:
            return self.model_metadata[style]
        
        metadata_path = self.model_dir / style.value / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
            
        return None
    

class ClusterManager:
    """Manages clusters and nodes in the federated learning framework."""
    
    def __init__(self, eco_mappings: Dict[str, List[str]]):
        """Initializes the ClusterManager with ECO code mappings."""
        self.clusters: Dict[str, ClusterInfo] = {}
        self.node_to_cluster: Dict[str, str] = {}
        
        # Initialize pre-defined clusters
        for style_name, eco_codes in eco_mappings.items():
            try:
                style = ChessStyle(style_name)
                cluster_id = f"{style.value}_cluster"
                
                self.clusters[cluster_id] = ClusterInfo(
                    cluster_id=cluster_id,
                    style=style,
                    eco_codes=eco_codes,
                    nodes={}
                )
                logging.info(f"Initialized cluster {cluster_id} for style {style.value}.")

            except ValueError:
                logging.error(f"Invalid chess style in ECO mappings: {style_name}")
                
    def register_node(self, node_id: str, style: ChessStyle, websocket) -> bool:
        """Registers a new node to the appropriate cluster based on its style."""
        cluster_id = f"{style.value}_cluster"
        
        if cluster_id not in self.clusters:
            logging.error(f"Cluster {cluster_id} does not exist.")
            return False
        
        node_info = NodeInfo(
            node_id=node_id,
            style=style,
            cluster_id=cluster_id,
            status=NodeStatus.CONNECTED,
            websocket=websocket,
            last_heartbeat=datetime.now(timezone.utc)
        )
        
        self.clusters[cluster_id].nodes[node_id] = node_info
        self.node_to_cluster[node_id] = cluster_id
        
        logging.info(f"Registered node {node_id} to cluster {cluster_id}.")
        return True
    
    def unregister_node(self, node_id: str):
        """Unregisters a node from its cluster."""
        if node_id in self.node_to_cluster:
            cluster_id = self.node_to_cluster[node_id]
            
            if cluster_id in self.clusters and node_id in self.clusters[cluster_id].nodes:
                del self.clusters[cluster_id].nodes[node_id]
                logging.info(f"Unregistered node {node_id} from cluster {cluster_id}.")
            
            del self.node_to_cluster[node_id]
        else:
            logging.warning(f"Node {node_id} not found in any cluster.")
            
    def get_cluster_nodes(self, cluster_id: str) -> List[NodeInfo]:
        """Returns the list of nodes in a given cluster."""
        if cluster_id in self.clusters:
            return list(self.clusters[cluster_id].nodes.values())
        return []
    
    def get_node_cluster(self, node_id: str) -> Optional[str]:
        """Returns the cluster ID for a given node."""
        return self.node_to_cluster.get(node_id, None)
    
    def update_node_status(self, node_id: str, status: NodeStatus):
        """Update a node's status."""
        if node_id in self.node_to_cluster:
            cluster_id = self.node_to_cluster[node_id]
            if cluster_id in self.clusters and node_id in self.clusters[cluster_id].nodes:
                self.clusters[cluster_id].nodes[node_id].status = status
                self.clusters[cluster_id].nodes[node_id].last_heartbeat = datetime.now(timezone.utc)
                logging.info(f"Updated status of node {node_id} to {status.value}.")
            else:
                logging.warning(f"Node {node_id} not found in cluster {cluster_id}.")
        else:
            logging.warning(f"Node {node_id} not found in any cluster.")
            

class FederatedServer:
    """Central server for managing federated learning nodes and clusters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.connected_clients: Set[WebSocketServerProtocol] = set()
        
        # Initialize components
        eco_mappings = config.get('clusters', {})
        self.cluster_manager = ClusterManager(eco_mappings)
        self.model_registry = CentralModelRegistry(config.get('model_storage', {}).get('path', 'models'))
        
        # Training state
        self.global_training_phase = TrainingPhase.INDIVIDUAL_DEVELOPMENT
        self.connected_nodes: Set[WebSocketServerProtocol] = set()
        
        # Threading for model operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("FederatedServer")
        
    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handles incoming client connections."""
        client_id = str(uuid.uuid4())
        self.connected_clients.add(websocket)
        self.logger.info(f"Client {client_id} connected from {websocket.remote_address}.")
        
        try:
            async for message in websocket:
                await self.process_message(client_id, websocket, message)
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected.")
        except Exception as e:
            self.logger.error(f"Error with client {client_id}: {e}")
        finally:
            self.connected_clients.discard(websocket)
            # Clean up node registration if it exists
            self.cluster_manager.unregister_node(client_id)
            
    async def process_message(self, client_id: str, websocket, raw_message: str):
        """Process incoming messages from nodes."""
        try:
            message = json.loads(raw_message)
            message_type = message.get('type')
            
            if message_type == 'register':
                await self.handle_node_registration(client_id, websocket, message)
            elif message_type == 'heartbeat':
                await self.handle_heartbeat(client_id, message)
            elif message_type == 'training_complete':
                await self.handle_training_complete(client_id, message)
            elif message_type == 'model_update':
                await self.handle_model_update(client_id, message)
            else:
                await self.send_error(websocket, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self.send_error(websocket, "Invalid JSON message")
        except Exception as e:
            self.logger.error(f"Error processing message from {client_id}: {e}")
            await self.send_error(websocket, f"Server error: {str(e)}")
    
    async def handle_node_registration(self, client_id: str, websocket, message: Dict):
        """Handle node registration to clusters."""
        try:
            style_str = str(message.get('style', '')).lower()
            style = ChessStyle(style_str)
            
            success = self.cluster_manager.register_node(client_id, style, websocket)
            
            if success:
                cluster_id = f"{style.value}_cluster"
                eco_codes = self.cluster_manager.clusters[cluster_id].eco_codes
                
                response = {
                    'type': 'registration_success',
                    'node_id': client_id,
                    'cluster_id': cluster_id,
                    'eco_codes': eco_codes,
                    'training_phase': self.global_training_phase.value
                }
                await websocket.send(json.dumps(response))
                
                self.logger.info(f"Node {client_id} registered to {cluster_id}")
            else:
                await self.send_error(websocket, "Registration failed")
                
        except ValueError:
            await self.send_error(websocket, f"Invalid chess style: {message.get('style')}")
        except Exception as e:
            await self.send_error(websocket, f"Registration error: {str(e)}")
            
    async def handle_heartbeat(self, client_id: str, message: Dict):
        """Handle heartbeat messages from nodes."""
        status_str = str(message.get('status', 'connected'))
        try:
            status = NodeStatus(status_str)
            self.cluster_manager.update_node_status(client_id, status)
        except ValueError:
            self.logger.warning(f"Invalid status from node {client_id}: {status_str}")
    
    async def handle_training_complete(self, client_id: str, message: Dict):
        """Handle training completion notifications from nodes."""
        cluster_id = self.cluster_manager.get_node_cluster(client_id)
        if not cluster_id:
            self.logger.warning(f"Training complete from unregistered node: {client_id}")
            return
        
        # Update node status
        self.cluster_manager.update_node_status(client_id, NodeStatus.READY_FOR_AGGREGATION)
        
        # Check if cluster is ready for aggregation
        await self.check_cluster_aggregation_readiness(cluster_id)
        
    async def handle_model_update(self, client_id: str, message: Dict):
        """Handle model parameter updates from nodes."""
        cluster_id = self.cluster_manager.get_node_cluster(client_id)
        if not cluster_id:
            self.logger.warning(f"Model update from unregistered node: {client_id}")
            return
        
        # Store model parameters (simplified - in practice would handle serialized model data)
        model_data = message.get('model_data')
        if model_data:
            self.logger.info(f"Received model update from node {client_id} in cluster {cluster_id}")
            # TODO: Store model data for aggregation
            
        # Update node status
        self.cluster_manager.update_node_status(client_id, NodeStatus.AGGREGATING)
        
    async def check_cluster_aggregation_readiness(self, cluster_id: str):
        """Check if a cluster is ready for aggregation."""
        cluster_nodes = self.cluster_manager.get_cluster_nodes(cluster_id)
        ready_nodes = [node for node in cluster_nodes if node.status == NodeStatus.READY_FOR_AGGREGATION]
        
        # Simple readiness criterion: at least 2 nodes ready
        if len(ready_nodes) >= 2:
            self.logger.info(f"Cluster {cluster_id} ready for aggregation with {len(ready_nodes)} nodes")
            await self.trigger_cluster_aggregation(cluster_id)
            
    async def trigger_cluster_aggregation(self, cluster_id: str):
        """Trigger aggregation for a specific cluster."""
        cluster_nodes = self.cluster_manager.get_cluster_nodes(cluster_id)
        ready_nodes = [node for node in cluster_nodes if node.status == NodeStatus.READY_FOR_AGGREGATION]
        
        # Send aggregation request to ready nodes
        aggregation_message = {
            'type': 'aggregation_request',
            'cluster_id': cluster_id,
            'participating_nodes': [node.node_id for node in ready_nodes]
        }
        
        for node in ready_nodes:
            if node.websocket:
                try:
                    await node.websocket.send(json.dumps(aggregation_message))
                    self.cluster_manager.update_node_status(node.node_id, NodeStatus.AGGREGATING)
                except Exception as e:
                    self.logger.error(f"Failed to send aggregation request to {node.node_id}: {e}")
                    
    async def broadcast_phase_transition(self, new_phase: TrainingPhase):
        """Broadcast training phase transition to all connected nodes."""
        message = {
            'type': 'phase_transition',
            'new_phase': new_phase.value,
            'timestamp': datetime.now().isoformat()
        }
        
        disconnected_clients = set()
        for websocket in self.connected_clients.copy():
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(websocket)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(websocket)
        
        # Clean up disconnected clients
        self.connected_clients -= disconnected_clients
        
        self.logger.info(f"Broadcasted phase transition to {new_phase.value}")
        
    async def send_error(self, websocket, error_message: str):
        """Send error message to a client."""
        try:
            error_response = {
                'type': 'error',
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            }
            await websocket.send(json.dumps(error_response))
        except Exception as e:
            self.logger.error(f"Failed to send error message: {e}")

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current status of all clusters."""
        status = {
            'global_phase': self.global_training_phase.value,
            'clusters': {},
            'total_nodes': 0
        }
        
        for cluster_id, cluster_info in self.cluster_manager.clusters.items():
            node_count = len(cluster_info.nodes)
            status['total_nodes'] += node_count
            
            status['clusters'][cluster_id] = {
                'style': cluster_info.style.value,
                'node_count': node_count,
                'eco_codes_count': len(cluster_info.eco_codes),
                'model_version': cluster_info.current_model_version,
                'training_phase': cluster_info.training_phase.value,
                'last_aggregation': cluster_info.last_aggregation.isoformat() if cluster_info.last_aggregation else None,
                'nodes': {
                    node_id: {
                        'status': node.status.value,
                        'last_heartbeat': node.last_heartbeat.isoformat() if node.last_heartbeat else None,
                        'model_version': node.model_version,
                        'training_iterations': node.training_iterations
                    }
                    for node_id, node in cluster_info.nodes.items()
                }
            }
        
        return status
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start the WebSocket server."""
        self.running = True
        self.logger.info(f"Starting FedRL server on {host}:{port}")
        
        # Start heartbeat monitoring task
        heartbeat_task = asyncio.create_task(self.heartbeat_monitor())
        
        # Start WebSocket server
        try:
            async with websockets.serve(self.handle_client, host, port):
                self.logger.info(f"FedRL server running on ws://{host}:{port}")
                
                # Keep server running
                while self.running:
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            heartbeat_task.cancel()
            self.executor.shutdown(wait=True)
    
    async def heartbeat_monitor(self):
        """Monitor node heartbeats and remove stale connections."""
        while self.running:
            try:
                current_time = datetime.now()
                stale_threshold = timedelta(minutes=5)  # 5 minute timeout
                
                stale_nodes = []
                for cluster_id, cluster_info in self.cluster_manager.clusters.items():
                    for node_id, node_info in cluster_info.nodes.items():
                        if (node_info.last_heartbeat and 
                            current_time - node_info.last_heartbeat > stale_threshold):
                            stale_nodes.append(node_id)
                
                # Remove stale nodes
                for node_id in stale_nodes:
                    self.logger.warning(f"Removing stale node: {node_id}")
                    self.cluster_manager.unregister_node(node_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(30)
    
    def stop_server(self):
        """Stop the server gracefully."""
        self.logger.info("Stopping FedRL server...")
        self.running = False
        
        
async def main():
    """Example server startup."""
    # Example configuration
    config = {
        'clusters': {
            'tactical': ['B20-B99', 'E60-E99', 'A80-A99'],     # Sicilian, King's Indian, Dutch
            'positional': ['D06-D69', 'A10-A39', 'B10-B19'],  # Queen's Gambit, English, Caro-Kann
            'dynamic': ['A04-A09', 'E00-E09', 'E20-E59']       # Réti, Catalan, Nimzo-Indian
        },
        'model_storage': {
            'path': './models',
            'versioning': True,
            'auto_backup': True
        }
    }
    
    server = FederatedServer(config)
    
    try:
        await server.start_server(host="0.0.0.0", port=8765)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop_server()


if __name__ == "__main__":
    asyncio.run(main())