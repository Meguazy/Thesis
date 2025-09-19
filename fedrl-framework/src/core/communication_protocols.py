#!/usr/bin/env python3
"""
FedRL Framework - Communication Protocols and Message Schemas

Defines all message types and data structures for node-server communication
in the clustered federated learning system.

Author: Based on Francesco Finucci's FedRL Framework
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import json
import uuid


class MessageType(Enum):
    """All supported message types in the FedRL protocol."""
    # Node registration and lifecycle
    REGISTER = "register"
    REGISTRATION_SUCCESS = "registration_success"
    REGISTRATION_FAILED = "registration_failed"
    
    # Heartbeat and status
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    
    # Training coordination
    PHASE_TRANSITION = "phase_transition"
    TRAINING_COMPLETE = "training_complete"
    START_TRAINING = "start_training"
    STOP_TRAINING = "stop_training"
    
    # Model aggregation
    AGGREGATION_REQUEST = "aggregation_request"
    MODEL_UPDATE = "model_update"
    AGGREGATED_MODEL = "aggregated_model"
    AGGREGATION_COMPLETE = "aggregation_complete"
    
    # Cluster management
    CLUSTER_STATUS = "cluster_status"
    CLUSTER_UPDATE = "cluster_update"
    
    # General
    ERROR = "error"
    ACK = "ack"
    PING = "ping"
    PONG = "pong"


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
    ERROR = "error"


@dataclass
class BaseMessage:
    """Base class for all protocol messages."""
    type: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    protocol_version: str = "1.0"
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(asdict(self), indent=None, separators=(',', ':'))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class NodeCapabilities:
    """Describes a node's training capabilities."""
    max_iterations_per_phase: int
    self_play_games_per_iteration: int
    supported_model_architectures: List[str]
    gpu_available: bool = False
    cpu_cores: int = 1
    memory_gb: float = 1.0


@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_game_length: float = 0.0
    training_loss: float = 0.0
    model_version: int = 0
    eco_adherence_rate: float = 0.0
    iterations_completed: int = 0
    training_time_minutes: float = 0.0


# === NODE REGISTRATION MESSAGES ===

@dataclass
class RegisterMessage(BaseMessage):
    """Node registration request."""
    type: str = MessageType.REGISTER.value
    node_id: str = ""
    style: str = ""  # ChessStyle enum value
    capabilities: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())


@dataclass
class RegistrationSuccessMessage(BaseMessage):
    """Successful node registration response."""
    type: str = MessageType.REGISTRATION_SUCCESS.value
    node_id: str = ""
    cluster_id: str = ""
    assigned_style: str = ""
    eco_codes: List[str] = field(default_factory=list)
    current_phase: str = TrainingPhase.INDIVIDUAL_DEVELOPMENT.value
    server_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationFailedMessage(BaseMessage):
    """Failed node registration response."""
    type: str = MessageType.REGISTRATION_FAILED.value
    node_id: str = ""
    error_code: str = ""
    error_message: str = ""
    retry_allowed: bool = True


# === HEARTBEAT AND STATUS MESSAGES ===

@dataclass
class HeartbeatMessage(BaseMessage):
    """Node heartbeat message."""
    type: str = MessageType.HEARTBEAT.value
    node_id: str = ""
    status: str = NodeStatus.CONNECTED.value
    metrics: Optional[Dict[str, Any]] = None
    current_iteration: int = 0
    phase: str = TrainingPhase.INDIVIDUAL_DEVELOPMENT.value


@dataclass
class StatusUpdateMessage(BaseMessage):
    """Node status update message."""
    type: str = MessageType.STATUS_UPDATE.value
    node_id: str = ""
    old_status: str = ""
    new_status: str = ""
    details: Optional[Dict[str, Any]] = None


# === TRAINING COORDINATION MESSAGES ===

@dataclass
class PhaseTransitionMessage(BaseMessage):
    """Training phase transition notification."""
    type: str = MessageType.PHASE_TRANSITION.value
    old_phase: str = ""
    new_phase: str = ""
    effective_time: str = ""
    transition_reason: str = ""
    cluster_specific: bool = False
    target_clusters: List[str] = field(default_factory=list)


@dataclass
class TrainingCompleteMessage(BaseMessage):
    """Node training completion notification."""
    type: str = MessageType.TRAINING_COMPLETE.value
    node_id: str = ""
    cluster_id: str = ""
    phase: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_checksum: str = ""


@dataclass
class StartTrainingMessage(BaseMessage):
    """Server instruction to start training."""
    type: str = MessageType.START_TRAINING.value
    node_id: str = ""
    phase: str = ""
    training_config: Dict[str, Any] = field(default_factory=dict)
    eco_constraints: List[str] = field(default_factory=list)


@dataclass
class StopTrainingMessage(BaseMessage):
    """Server instruction to stop training."""
    type: str = MessageType.STOP_TRAINING.value
    node_id: str = ""
    reason: str = ""
    save_progress: bool = True


# === MODEL AGGREGATION MESSAGES ===

@dataclass
class AggregationRequestMessage(BaseMessage):
    """Server request for model aggregation."""
    type: str = MessageType.AGGREGATION_REQUEST.value
    aggregation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id: str = ""
    participating_nodes: List[str] = field(default_factory=list)
    aggregation_type: str = "intra_cluster"  # or "inter_cluster"
    deadline: str = ""


@dataclass
class ModelUpdateMessage(BaseMessage):
    """Node model parameters for aggregation."""
    type: str = MessageType.MODEL_UPDATE.value
    node_id: str = ""
    cluster_id: str = ""
    aggregation_id: str = ""
    model_data: Dict[str, Any] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    data_size: int = 0  # Number of training samples used


@dataclass
class AggregatedModelMessage(BaseMessage):
    """Server sending aggregated model to nodes."""
    type: str = MessageType.AGGREGATED_MODEL.value
    aggregation_id: str = ""
    cluster_id: str = ""
    target_nodes: List[str] = field(default_factory=list)
    model_data: Dict[str, Any] = field(default_factory=dict)
    model_version: int = 1
    aggregation_method: str = "fedavg"
    participating_node_count: int = 0
    aggregation_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class AggregationCompleteMessage(BaseMessage):
    """Confirmation that aggregation is complete."""
    type: str = MessageType.AGGREGATION_COMPLETE.value
    aggregation_id: str = ""
    cluster_id: str = ""
    success: bool = True
    participating_nodes: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


# === CLUSTER MANAGEMENT MESSAGES ===

@dataclass
class ClusterStatusMessage(BaseMessage):
    """Cluster status information."""
    type: str = MessageType.CLUSTER_STATUS.value
    cluster_id: str = ""
    style: str = ""
    node_count: int = 0
    active_nodes: int = 0
    current_phase: str = ""
    model_version: int = 1
    last_aggregation: str = ""
    eco_codes: List[str] = field(default_factory=list)
    performance_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterUpdateMessage(BaseMessage):
    """Cluster configuration update."""
    type: str = MessageType.CLUSTER_UPDATE.value
    cluster_id: str = ""
    updates: Dict[str, Any] = field(default_factory=dict)
    affected_nodes: List[str] = field(default_factory=list)


# === GENERAL MESSAGES ===

@dataclass
class ErrorMessage(BaseMessage):
    """Error message."""
    type: str = MessageType.ERROR.value
    error_code: str = ""
    error_message: str = ""
    details: Optional[Dict[str, Any]] = None
    recoverable: bool = False
    suggested_action: str = ""


@dataclass
class AckMessage(BaseMessage):
    """Acknowledgment message."""
    type: str = MessageType.ACK.value
    ack_message_id: str = ""
    success: bool = True
    details: Optional[Dict[str, Any]] = None


@dataclass
class PingMessage(BaseMessage):
    """Ping message for connection testing."""
    type: str = MessageType.PING.value


@dataclass
class PongMessage(BaseMessage):
    """Pong response to ping."""
    type: str = MessageType.PONG.value
    ping_message_id: str = ""


# === MESSAGE FACTORY ===

class MessageFactory:
    """Factory for creating protocol messages."""
    
    MESSAGE_CLASSES = {
        MessageType.REGISTER.value: RegisterMessage,
        MessageType.REGISTRATION_SUCCESS.value: RegistrationSuccessMessage,
        MessageType.REGISTRATION_FAILED.value: RegistrationFailedMessage,
        MessageType.HEARTBEAT.value: HeartbeatMessage,
        MessageType.STATUS_UPDATE.value: StatusUpdateMessage,
        MessageType.PHASE_TRANSITION.value: PhaseTransitionMessage,
        MessageType.TRAINING_COMPLETE.value: TrainingCompleteMessage,
        MessageType.START_TRAINING.value: StartTrainingMessage,
        MessageType.STOP_TRAINING.value: StopTrainingMessage,
        MessageType.AGGREGATION_REQUEST.value: AggregationRequestMessage,
        MessageType.MODEL_UPDATE.value: ModelUpdateMessage,
        MessageType.AGGREGATED_MODEL.value: AggregatedModelMessage,
        MessageType.AGGREGATION_COMPLETE.value: AggregationCompleteMessage,
        MessageType.CLUSTER_STATUS.value: ClusterStatusMessage,
        MessageType.CLUSTER_UPDATE.value: ClusterUpdateMessage,
        MessageType.ERROR.value: ErrorMessage,
        MessageType.ACK.value: AckMessage,
        MessageType.PING.value: PingMessage,
        MessageType.PONG.value: PongMessage,
    }
    
    @classmethod
    def create_message(cls, message_type: str, **kwargs) -> BaseMessage:
        """Create a message of the specified type."""
        message_class = cls.MESSAGE_CLASSES.get(message_type)
        if message_class is None:
            raise ValueError(f"Unknown message type: {message_type}")
        
        return message_class(**kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> BaseMessage:
        """Create a message from JSON string."""
        try:
            data = json.loads(json_str)
            message_type = data.get('type')
            
            if message_type not in cls.MESSAGE_CLASSES:
                raise ValueError(f"Unknown message type: {message_type}")
            
            message_class = cls.MESSAGE_CLASSES[message_type]
            return message_class(**data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except Exception as e:
            raise ValueError(f"Error creating message: {e}")
    
    @classmethod
    def validate_message(cls, message: BaseMessage) -> bool:
        """Validate a message structure."""
        try:
            # Check required fields
            if not hasattr(message, 'type') or not message.type:
                return False
            
            if not hasattr(message, 'message_id') or not message.message_id:
                return False
            
            # Validate message type
            if message.type not in [mt.value for mt in MessageType]:
                return False
            
            return True
            
        except Exception:
            return False


# === PROTOCOL UTILITIES ===

class ProtocolError(Exception):
    """Base exception for protocol errors."""
    pass


class MessageValidationError(ProtocolError):
    """Exception for message validation errors."""
    pass


class UnsupportedMessageTypeError(ProtocolError):
    """Exception for unsupported message types."""
    pass


def create_error_message(error_code: str, error_message: str, 
                        recoverable: bool = False, 
                        suggested_action: str = "") -> ErrorMessage:
    """Helper function to create error messages."""
    return ErrorMessage(
        error_code=error_code,
        error_message=error_message,
        recoverable=recoverable,
        suggested_action=suggested_action
    )


def create_ack_message(original_message: BaseMessage, 
                      success: bool = True, 
                      details: Optional[Dict[str, Any]] = None) -> AckMessage:
    """Helper function to create acknowledgment messages."""
    return AckMessage(
        ack_message_id=original_message.message_id,
        success=success,
        details=details
    )


# === EXAMPLE USAGE ===

if __name__ == "__main__":
    # Example of creating and using messages
    
    # Create registration message
    reg_msg = RegisterMessage(
        node_id="node_123",
        style="tactical",
        capabilities={
            "max_iterations": 100,
            "gpu_available": True
        }
    )
    
    # Convert to JSON
    json_str = reg_msg.to_json()
    print("Registration message JSON:")
    print(json_str)
    print()
    
    # Parse back from JSON
    parsed_msg = MessageFactory.from_json(json_str)
    print(f"Parsed message type: {parsed_msg.type}")
    print(f"Node ID: {parsed_msg.node_id}")
    print()
    
    # Create success response
    success_msg = RegistrationSuccessMessage(
        node_id="node_123",
        cluster_id="tactical_cluster",
        assigned_style="tactical",
        eco_codes=["B20-B99", "E60-E99"],
        current_phase="individual_development"
    )
    
    print("Registration success message:")
    print(success_msg.to_json())