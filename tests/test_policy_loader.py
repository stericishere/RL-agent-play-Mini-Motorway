"""Tests for policy loading functionality."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from motorways.policy.action_space import decode_action, encode_action, get_action_space_size, sample_random_action
from motorways.policy.loader import create_random_policy, get_device_recommendation
from motorways.config.schema import Action


class TestActionSpace:
    """Test action space encoding/decoding."""
    
    def test_decode_action_noop(self):
        """Test decoding no-op action."""
        action = decode_action(0, 10, 10)
        
        assert action.type == "noop"
        assert action.r is None
        assert action.c is None
    
    def test_decode_action_click(self):
        """Test decoding click actions."""
        # Test first click action (grid position 0,0)
        action = decode_action(1, 10, 10)
        
        assert action.type == "click"
        assert action.r == 0
        assert action.c == 0
        
        # Test last click action (grid position 9,9)  
        action = decode_action(100, 10, 10)  # 10*10 = 100th action
        
        assert action.type == "click"
        assert action.r == 9
        assert action.c == 9
    
    def test_decode_action_drag(self):
        """Test decoding drag actions."""
        # Test first drag action
        action = decode_action(101, 10, 10)  # After 100 click actions + 1
        
        assert action.type == "drag"
        assert action.path is not None
        assert len(action.path) >= 2
        assert action.path[0] == (0, 0)  # Should start at grid 0,0
    
    def test_decode_action_toolbar(self):
        """Test decoding toolbar actions."""
        action_value = 2 * 10 * 10 + 1  # After clicks and drags + 1
        action = decode_action(action_value, 10, 10)
        
        assert action.type == "toolbar"
        assert action.tool == "road"
        
        # Test other toolbar actions
        action = decode_action(action_value + 1, 10, 10)
        assert action.tool == "bridge"
    
    def test_decode_action_invalid(self):
        """Test decoding invalid action values."""
        # Very large action value should return noop
        action = decode_action(99999, 10, 10)
        assert action.type == "noop"
        
        # Negative action value should return noop  
        action = decode_action(-1, 10, 10)
        assert action.type == "noop"
    
    def test_encode_action_noop(self):
        """Test encoding no-op action."""
        action = Action(type="noop")
        encoded = encode_action(action, 10, 10)
        
        assert encoded == 0
    
    def test_encode_action_click(self):
        """Test encoding click actions."""
        action = Action(type="click", r=0, c=0)
        encoded = encode_action(action, 10, 10)
        
        assert encoded == 1
        
        action = Action(type="click", r=9, c=9)
        encoded = encode_action(action, 10, 10)
        
        assert encoded == 100  # 9*10 + 9 + 1
    
    def test_encode_action_click_invalid_coords(self):
        """Test encoding click action with invalid coordinates."""
        action = Action(type="click", r=10, c=0)  # Out of bounds
        
        with pytest.raises(ValueError):
            encode_action(action, 10, 10)
        
        action = Action(type="click", r=0, c=10)  # Out of bounds
        
        with pytest.raises(ValueError):
            encode_action(action, 10, 10)
    
    def test_encode_action_drag(self):
        """Test encoding drag actions."""
        action = Action(type="drag", path=[(0, 0), (0, 1)])
        encoded = encode_action(action, 10, 10)
        
        # Should encode based on starting position
        expected = 10 * 10 + 1  # After all click actions + 1
        assert encoded == expected
    
    def test_encode_action_toolbar(self):
        """Test encoding toolbar actions."""
        action = Action(type="toolbar", tool="road")
        encoded = encode_action(action, 10, 10)
        
        expected = 2 * 10 * 10 + 1  # After clicks + drags + 1
        assert encoded == expected
        
        action = Action(type="toolbar", tool="bridge")
        encoded = encode_action(action, 10, 10)
        
        expected = 2 * 10 * 10 + 2
        assert encoded == expected
    
    def test_encode_decode_roundtrip(self):
        """Test that encode/decode are inverse operations."""
        grid_h, grid_w = 8, 12
        
        # Test various action types
        test_actions = [
            Action(type="noop"),
            Action(type="click", r=0, c=0),
            Action(type="click", r=7, c=11),
            Action(type="drag", path=[(3, 4), (3, 5)]),
            Action(type="toolbar", tool="road"),
            Action(type="toolbar", tool="bridge")
        ]
        
        for original_action in test_actions:
            encoded = encode_action(original_action, grid_h, grid_w)
            decoded = decode_action(encoded, grid_h, grid_w)
            
            assert decoded.type == original_action.type
            
            if original_action.type == "click":
                assert decoded.r == original_action.r
                assert decoded.c == original_action.c
            elif original_action.type == "toolbar":
                assert decoded.tool == original_action.tool
    
    def test_get_action_space_size(self):
        """Test action space size calculation."""
        size = get_action_space_size(10, 10)
        expected = 1 + 100 + 100 + 6  # noop + clicks + drags + toolbar
        assert size == expected
        
        size = get_action_space_size(5, 8)
        expected = 1 + 40 + 40 + 6
        assert size == expected
    
    def test_sample_random_action(self):
        """Test random action sampling."""
        action = sample_random_action(10, 10)
        
        # Should return a valid Action object
        assert isinstance(action, Action)
        assert action.type in ["noop", "click", "drag", "toolbar"]
        
        # Test multiple samples to ensure variety
        action_types = set()
        for _ in range(50):
            action = sample_random_action(10, 10)
            action_types.add(action.type)
        
        # Should see multiple action types with enough samples
        assert len(action_types) > 1
    
    def test_sample_random_action_with_mask(self):
        """Test random action sampling with mask."""
        mask = np.ones(get_action_space_size(10, 10), dtype=bool)
        mask[0] = False  # Disable noop
        
        # Sample many actions
        for _ in range(20):
            action = sample_random_action(10, 10, mask)
            assert action.type != "noop"  # Should not sample disabled action
    
    def test_sample_random_action_empty_mask(self):
        """Test random action sampling with empty mask."""
        mask = np.zeros(get_action_space_size(10, 10), dtype=bool)
        
        action = sample_random_action(10, 10, mask)
        assert action.type == "noop"  # Fallback when no valid actions


class TestPolicyLoader:
    """Test policy loading functionality."""
    
    def test_create_random_policy(self):
        """Test creating random policy."""
        policy = create_random_policy(16, 16)
        
        # Should return a callable
        assert callable(policy)
        
        # Test calling the policy
        dummy_obs = np.random.rand(1, 3, 64, 64)
        action = policy(dummy_obs)
        
        assert isinstance(action, Action)
        assert action.type in ["noop", "click", "drag", "toolbar"]
    
    def test_get_device_recommendation_no_torch(self):
        """Test device recommendation when torch is not available."""
        with patch('motorways.policy.loader.torch', None):
            device = get_device_recommendation()
            assert device == "cpu"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_device_recommendation_mps(self, mock_cuda, mock_mps):
        """Test device recommendation when MPS is available."""
        mock_mps.return_value = True
        mock_cuda.return_value = False
        
        device = get_device_recommendation()
        assert device == "mps"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_device_recommendation_cuda(self, mock_cuda, mock_mps):
        """Test device recommendation when CUDA is available."""
        mock_mps.return_value = False
        mock_cuda.return_value = True
        
        device = get_device_recommendation()
        assert device == "cuda"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_device_recommendation_cpu(self, mock_cuda, mock_mps):
        """Test device recommendation when only CPU is available."""
        mock_mps.return_value = False
        mock_cuda.return_value = False
        
        device = get_device_recommendation()
        assert device == "cpu"


class TestActionValidation:
    """Test action validation and edge cases."""
    
    def test_action_creation_invalid_click(self):
        """Test Action creation with invalid click parameters."""
        with pytest.raises(ValueError):
            Action(type="click", r=None, c=0)
        
        with pytest.raises(ValueError):
            Action(type="click", r=0, c=None)
    
    def test_action_creation_invalid_drag(self):
        """Test Action creation with invalid drag parameters."""
        with pytest.raises(ValueError):
            Action(type="drag", path=None)
        
        with pytest.raises(ValueError):
            Action(type="drag", path=[(0, 0)])  # Need at least 2 points
        
        with pytest.raises(ValueError):
            Action(type="drag", path=[])
    
    def test_action_creation_invalid_toolbar(self):
        """Test Action creation with invalid toolbar parameters."""
        with pytest.raises(ValueError):
            Action(type="toolbar", tool=None)
    
    def test_action_creation_invalid_type(self):
        """Test Action creation with invalid type."""
        with pytest.raises(ValueError):
            Action(type="invalid_type")
    
    def test_action_to_dict(self):
        """Test Action to_dict conversion."""
        action = Action(type="click", r=5, c=7)
        result = action.to_dict()
        
        expected = {
            "type": "click",
            "r": 5,
            "c": 7,
            "path": None,
            "tool": None
        }
        assert result == expected
        
        action = Action(type="drag", path=[(0, 1), (2, 3)])
        result = action.to_dict()
        
        expected = {
            "type": "drag", 
            "r": None,
            "c": None,
            "path": [(0, 1), (2, 3)],
            "tool": None
        }
        assert result == expected