"""
FedRL Framework - Federated Learning Node Implementation

This implements individual training nodes that participate in clustered
federated learning with ECO opening constraints.

Author: Based on Francesco Finucci's FedRL Framework
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import websockets
import torch
import threading
from pathlib import Path
import chess
import random


class TrainingPhase(Enum):
    """Training phases in the clustered federated learning framework."""
    INDIVIDUAL_DEVELOPMENT = "individual_development"
    INTRA_CLUSTER = "intra_cluster"
    INTER_CLUSTER = "inter_cluster"
    COMPLETED = "completed"


class ChessStyle(Enum):
    """Chess playing styles based on ECO opening constraints."""
    TACTICAL = "tactical"
    POSITIONAL = "positional"
    DYNAMIC = "dynamic"


class NodeStatus(Enum):
    """Node training status."""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    TRAINING = "training"
    READY_FOR_AGGREGATION = "ready_for_aggregation"
    AGGREGATING = "aggregating"
    WAITING = "waiting"


@dataclass
class TrainingConfig:
    """"Configuration for training phases and parameters."""
    iterations_per_phase: int = 100
    eco_codes: List[str] = None
    self_play_games: int = 25
    batch_size: int = 32
    learning_rate: float = 0.001
    model_save_frequency: int = 10  # Save model every n iterations
    
    def __post_init__(self):
        if self.eco_codes is None:
            self.eco_codes = []


@dataclass
class TrainingMetrics:
    """Metrics to track training progress."""
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    avg_game_length: float = 0.0
    training_loss: float = 0.0
    model_version: int = 0
    eco_adherence_rate: float = 0.0
    
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "avg_game_length": self.avg_game_length,
            "training_loss": self.training_loss,
            "model_version": self.model_version,
            "eco_adherence_rate": self.eco_adherence_rate,
            "win_rate": self.win_rate(),
        }
        
        
class ECOConstraintManager:
    """Manages ECO opening constraints for chess games."""
    
    def __init__(self, eco_codes: List[str]):
        self.eco_codes = eco_codes
        self.opening_moves = self._load_opening_moves()
    
    def _load_opening_moves(self) -> Dict[str, List[str]]:
        """
        Load opening moves for ECO codes.

        TODO: This is a simplified version. In practice, you'd load from a proper ECO database.
        """
        # Simplified ECO opening moves mapping
        eco_openings = {
            # Tactical openings
            'B20': ['e4', 'c5'],  # Sicilian Defense
            'B21': ['e4', 'c5', 'f4'],  # Sicilian, Grand Prix Attack
            'E60': ['d4', 'Nf6', 'c4', 'g6'],  # King's Indian Defense
            'E61': ['d4', 'Nf6', 'c4', 'g6', 'Nc3'],  # King's Indian Defense
            'A80': ['d4', 'f5'],  # Dutch Defense
            
            # Positional openings
            'D06': ['d4', 'd5', 'c4'],  # Queen's Gambit
            'D07': ['d4', 'd5', 'c4', 'Nc6'],  # Queen's Gambit Declined
            'A10': ['c4'],  # English Opening
            'A15': ['c4', 'Nf6'],  # English Opening, Anglo-Indian Defense
            'B10': ['e4', 'c6'],  # Caro-Kann Defense
            
            # Dynamic openings
            'A04': ['Nf3'],  # Réti Opening
            'A05': ['Nf3', 'Nf6'],  # Réti Opening
            'E00': ['d4', 'Nf6', 'c4', 'e6', 'g3'],  # Catalan Opening
            'E20': ['d4', 'Nf6', 'c4', 'e6', 'Nc3', 'Bb4'],  # Nimzo-Indian Defense
        }
        
        # Filter to only requested ECO codes
        filtered_openings = {}
        for eco_code in self.eco_codes:
            # Handle ranges like B20-B99
            if '-' in eco_code:
                start_code, end_code = eco_code.split('-')
                # Simplified range handling - in practice, use proper ECO parsing
                for code in eco_openings:
                    if start_code <= code <= end_code:
                        filtered_openings[code] = eco_openings[code]
            elif eco_code in eco_openings:
                filtered_openings[eco_code] = eco_openings[eco_code]
        
        return filtered_openings
    
    def get_random_opening(self) -> List[str]:
        """Get a random opening sequence from the available ECO codes."""
        if not self.opening_moves:
            return []
        eco_code = random.choice(list(self.opening_moves.keys()))
        return self.opening_moves[eco_code]
    
    def validate_opening(self, moves: List[str]) -> bool:
        """Validate if the given moves adhere to any of the ECO openings."""
        for eco_moves in self.opening_moves.values():
            if len(moves) >= len(eco_moves):
                if moves[:len(eco_moves)] == eco_moves:
                    return True
        return False
    

class ChessEngineInterface:
    """
    Interface to chess engine for training.
    
    This is a placeholder interface. In practice, this would integrate
    with your existing chess engine repository.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.model = None # Placeholder for the chess model
        
    def load_model(self, model_path: str):
        """Load a chess model from the specified path."""
        self.model_path = Path(model_path)
        # TODO: Implement actual model loading logic
        logging.info(f"Loaded model from {model_path}")
        
    def save_model(self, model_path: str):
        """Save the current chess model to the specified path."""
        self.model_path = Path(model_path)
        # TODO: Implement actual model saving logic
        logging.info(f"Saved model to {model_path}")

    def train_iteration(self, eco_constraint: ECOConstraintManager, 
                       games_count: int = 25) -> TrainingMetrics:
        """Run a training iteration with ECO constraints.
        
        This is a simulation. In practice, it would:
        1. Generate self-play games using ECO-constrained openings
        2. Train the neural network on generated data
        3. Return training metrics
        """
        metrics = TrainingMetrics()
        
        # Simulate training
        time.sleep(0.5)  # Simulate training time
        
        # Simulate game results
        metrics.games_played = games_count
        metrics.wins = random.randint(games_count // 4, games_count // 2)
        metrics.draws = random.randint(games_count // 4, games_count // 2)
        metrics.losses = games_count - metrics.wins - metrics.draws
        metrics.avg_game_length = random.uniform(35.0, 65.0)
        metrics.training_loss = random.uniform(0.1, 0.5)
        metrics.eco_adherence_rate = random.uniform(0.85, 1.0)  # High adherence
        
        logging.info(f"Training iteration: {games_count} games, "
                    f"win rate: {metrics.win_rate():.2f}, "
                    f"ECO adherence: {metrics.eco_adherence_rate:.2f}")
        
        return metrics
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters for federated aggregation."""
        # TODO: Return actual model parameters
        return {"dummy_params": "placeholder"}
    
    def update_model_parameters(self, parameters: Dict[str, Any]):
        """Update model with aggregated parameters."""
        # TODO: Update actual model parameters
        logging.info("Updated model with aggregated parameters")
        

class FederatedNode:
    """Main federated learning node implementation."""
    
    def __init__(self, node_config: Dict[str, Any]):
        self.config = node_config
        self.node_id = str(uuid.uuid4())
        self.style = ChessStyle(node_config['style'])
        self.server_url = node_config.get('server_url', 'ws://localhost:8765')
        
        # Training components
        self.training_config = TrainingConfig(**node_config.get('training', {}))
        self.chess_engine = ChessEngineInterface(node_config.get('model_path'))
        self.eco_manager = None  # Will be set after server registration
        
        # Connection and state
        self.websocket = None
        self.running = False
        self.current_phase = TrainingPhase.INDIVIDUAL_DEVELOPMENT
        self.cluster_id = None
        
        # Metrics and logging
        self.training_metrics = TrainingMetrics()
        self.logger = logging.getLogger(f"Node-{self.style.value}")
        self.logger.setLevel(logging.INFO)

    async def connect_to_server(self) -> bool:
        """Connect to the federated learning server."""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.logger.info(f"Connected to server at {self.server_url}")
            
            # Register with server
            await self.register_with_server()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {e}")
            return False
    
    async def register_with_server(self):
        """Register this node with the server."""
        registration_message = {
            'type': 'register',
            'style': self.style.value,
            'node_id': self.node_id,
            'capabilities': {
                'training_config': {
                    'iterations_per_phase': self.training_config.iterations_per_phase,
                    'self_play_games': self.training_config.self_play_games
                }
            }
        }
        
        await self.websocket.send(json.dumps(registration_message))
        self.logger.info(f"Sent registration request for {self.style.value} style")
    
    async def handle_server_messages(self):
        """Handle messages from the server."""
        try:
            async for message in self.websocket:
                await self.process_server_message(message)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Server connection closed")
        except Exception as e:
            self.logger.error(f"Error handling server messages: {e}")
    
    async def process_server_message(self, raw_message: str):
        """Process individual messages from the server."""
        try:
            message = json.loads(raw_message)
            message_type = message.get('type')
            
            if message_type == 'registration_success':
                await self.handle_registration_success(message)
            elif message_type == 'phase_transition':
                await self.handle_phase_transition(message)
            elif message_type == 'aggregation_request':
                await self.handle_aggregation_request(message)
            elif message_type == 'aggregated_model':
                await self.handle_aggregated_model(message)
            elif message_type == 'error':
                self.logger.error(f"Server error: {message.get('message')}")
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            self.logger.error("Received invalid JSON from server")
        except Exception as e:
            self.logger.error(f"Error processing server message: {e}")
    
    async def handle_registration_success(self, message: Dict):
        """Handle successful registration with server."""
        self.cluster_id = message.get('cluster_id')
        eco_codes = message.get('eco_codes', [])
        self.current_phase = TrainingPhase(message.get('training_phase', 'individual_development'))
        
        # Initialize ECO constraint manager
        self.eco_manager = ECOConstraintManager(eco_codes)
        
        self.logger.info(f"Registration successful! Assigned to {self.cluster_id}")
        self.logger.info(f"ECO codes: {eco_codes}")
        self.logger.info(f"Current phase: {self.current_phase.value}")
        
        # Start training
        await self.start_training()
    
    async def handle_phase_transition(self, message: Dict):
        """Handle training phase transitions."""
        new_phase = TrainingPhase(message.get('new_phase'))
        self.logger.info(f"Phase transition: {self.current_phase.value} → {new_phase.value}")
        
        self.current_phase = new_phase
        
        # Adapt training for new phase
        if new_phase == TrainingPhase.INTRA_CLUSTER:
            self.logger.info("Entering intra-cluster training phase")
        elif new_phase == TrainingPhase.INTER_CLUSTER:
            self.logger.info("Entering inter-cluster training phase")
        elif new_phase == TrainingPhase.COMPLETED:
            self.logger.info("Training completed!")
            self.running = False
    
    async def handle_aggregation_request(self, message: Dict):
        """Handle aggregation requests from server."""
        participating_nodes = message.get('participating_nodes', [])
        self.logger.info(f"Aggregation requested with nodes: {participating_nodes}")
        
        # Send model parameters to server
        model_params = self.chess_engine.get_model_parameters()
        
        response = {
            'type': 'model_update',
            'node_id': self.node_id,
            'cluster_id': self.cluster_id,
            'model_data': model_params,
            'metrics': self.training_metrics.to_dict()
        }
        
        await self.websocket.send(json.dumps(response))
        self.logger.info("Sent model parameters for aggregation")
    
    async def handle_aggregated_model(self, message: Dict):
        """Handle receiving aggregated model from server."""
        aggregated_params = message.get('model_data')
        self.training_metrics.model_version = message.get('version', self.training_metrics.model_version + 1)
        
        # Update local model with aggregated parameters
        self.chess_engine.update_model_parameters(aggregated_params)
        
        self.logger.info(f"Updated model to version {self.training_metrics.model_version}")
        
        # Resume training
        await self.start_training()
    
    async def start_training(self):
        """Start the training process."""
        if not self.eco_manager:
            self.logger.error("ECO manager not initialized - cannot start training")
            return
        
        # Start training in a separate thread to avoid blocking message handling
        training_thread = threading.Thread(target=self.run_training_loop)
        training_thread.daemon = True
        training_thread.start()
    
    def run_training_loop(self):
        """Main training loop (runs in separate thread)."""
        self.logger.info("Starting training loop")
        
        for iteration in range(self.training_config.iterations_per_phase):
            if not self.running:
                break
            
            self.logger.info(f"Training iteration {iteration + 1}/{self.training_config.iterations_per_phase}")
            
            # Run training iteration with ECO constraints
            iteration_metrics = self.chess_engine.train_iteration(
                self.eco_manager, 
                self.training_config.self_play_games
            )
            
            # Update cumulative metrics
            self.training_metrics.games_played += iteration_metrics.games_played
            self.training_metrics.wins += iteration_metrics.wins
            self.training_metrics.draws += iteration_metrics.draws
            self.training_metrics.losses += iteration_metrics.losses
            self.training_metrics.training_loss = iteration_metrics.training_loss
            self.training_metrics.eco_adherence_rate = iteration_metrics.eco_adherence_rate
            
            # Send heartbeat with status
            asyncio.run_coroutine_threadsafe(
                self.send_heartbeat(NodeStatus.TRAINING), 
                asyncio.get_event_loop()
            )
            
            # Save model periodically
            if (iteration + 1) % self.training_config.model_save_frequency == 0:
                model_path = f"checkpoints/{self.node_id}_iteration_{iteration + 1}.pth"
                self.chess_engine.save_model(model_path)
        
        # Training phase complete
        self.logger.info(f"Training phase complete. Final metrics: {self.training_metrics.to_dict()}")
        
        # Notify server that training is complete
        asyncio.run_coroutine_threadsafe(
            self.send_training_complete(), 
            asyncio.get_event_loop()
        )
    
    async def send_heartbeat(self, status: NodeStatus):
        """Send heartbeat to server."""
        if self.websocket:
            try:
                heartbeat = {
                    'type': 'heartbeat',
                    'node_id': self.node_id,
                    'status': status.value,
                    'metrics': self.training_metrics.to_dict(),
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.websocket.send(json.dumps(heartbeat))
            except Exception as e:
                self.logger.error(f"Failed to send heartbeat: {e}")
    
    async def send_training_complete(self):
        """Notify server that training phase is complete."""
        if self.websocket:
            try:
                message = {
                    'type': 'training_complete',
                    'node_id': self.node_id,
                    'cluster_id': self.cluster_id,
                    'metrics': self.training_metrics.to_dict(),
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.websocket.send(json.dumps(message))
                await self.send_heartbeat(NodeStatus.READY_FOR_AGGREGATION)
                
            except Exception as e:
                self.logger.error(f"Failed to send training complete: {e}")
    
    async def run(self):
        """Main node execution loop."""
        self.running = True
        
        # Connect to server
        if not await self.connect_to_server():
            return
        
        # Start message handling
        try:
            await self.handle_server_messages()
        except KeyboardInterrupt:
            self.logger.info("Node stopped by user")
        except Exception as e:
            self.logger.error(f"Node error: {e}")
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()


async def main():
    """Example node startup."""
    node_config = {
        'style': 'tactical',  # or 'positional', 'dynamic'
        'server_url': 'ws://localhost:8765',
        'model_path': './initial_model.pth',
        'training': {
            'iterations_per_phase': 50,
            'self_play_games': 25,
            'batch_size': 32,
            'learning_rate': 0.001,
            'model_save_frequency': 10
        }
    }
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    node = FederatedNode(node_config)
    
    try:
        await node.run()
    except KeyboardInterrupt:
        print("\nShutting down node...")


if __name__ == "__main__":
    asyncio.run(main())