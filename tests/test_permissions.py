"""Tests for macOS permissions checking."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from motorways.utils.permissions import (
    check_screen_recording_permission,
    check_accessibility_permission, 
    check_all_permissions,
    get_permission_instructions
)


class TestPermissions:
    """Test permission checking functions."""
    
    @patch('motorways.utils.permissions.find_window')
    @patch('motorways.utils.permissions.grab_window')
    def test_check_screen_recording_permission_granted(self, mock_grab_window, mock_find_window):
        """Test screen recording permission check when granted."""
        # Mock successful window finding and capture
        mock_find_window.return_value = (12345, {'X': 0, 'Y': 0, 'Width': 800, 'Height': 600})
        mock_grab_window.return_value = Mock()
        mock_grab_window.return_value.size = 1000  # Non-empty image
        
        result = check_screen_recording_permission()
        
        assert result is True
        mock_find_window.assert_called_once_with("Finder")
        mock_grab_window.assert_called_once_with(12345)
    
    @patch('motorways.utils.permissions.find_window')
    def test_check_screen_recording_permission_no_window(self, mock_find_window):
        """Test screen recording permission check when no window found."""
        # Mock no window found initially, then fallback to window list
        mock_find_window.return_value = (None, None)
        
        with patch('motorways.utils.permissions.CGWindowListCopyWindowInfo') as mock_window_list:
            # Mock empty window list
            mock_window_list.return_value = []
            
            result = check_screen_recording_permission()
            
            assert result is False
    
    @patch('motorways.utils.permissions.find_window')
    @patch('motorways.utils.permissions.grab_window')
    def test_check_screen_recording_permission_capture_fails(self, mock_grab_window, mock_find_window):
        """Test screen recording permission check when capture fails."""
        mock_find_window.return_value = (12345, {'X': 0, 'Y': 0, 'Width': 800, 'Height': 600})
        mock_grab_window.side_effect = Exception("Permission denied")
        
        result = check_screen_recording_permission()
        
        assert result is False
    
    @patch('pyautogui.position')
    def test_check_accessibility_permission_granted(self, mock_position):
        """Test accessibility permission check when granted."""
        mock_position.return_value = Mock(x=100, y=200)
        
        result = check_accessibility_permission()
        
        assert result is True
        mock_position.assert_called_once()
    
    @patch('pyautogui.position')
    def test_check_accessibility_permission_denied(self, mock_position):
        """Test accessibility permission check when denied."""
        mock_position.side_effect = Exception("Permission denied")
        
        result = check_accessibility_permission()
        
        assert result is False
    
    @patch('pyautogui.position')
    def test_check_accessibility_permission_no_position(self, mock_position):
        """Test accessibility permission check when position is None."""
        mock_position.return_value = None
        
        result = check_accessibility_permission()
        
        assert result is False
    
    @patch('motorways.utils.permissions.check_screen_recording_permission')
    @patch('motorways.utils.permissions.check_accessibility_permission')
    def test_check_all_permissions_both_granted(self, mock_accessibility, mock_screen_recording):
        """Test checking all permissions when both are granted."""
        mock_screen_recording.return_value = True
        mock_accessibility.return_value = True
        
        result = check_all_permissions()
        
        expected = {
            "screen_recording": True,
            "accessibility": True
        }
        assert result == expected
    
    @patch('motorways.utils.permissions.check_screen_recording_permission')
    @patch('motorways.utils.permissions.check_accessibility_permission')
    def test_check_all_permissions_partial(self, mock_accessibility, mock_screen_recording):
        """Test checking all permissions when only some are granted."""
        mock_screen_recording.return_value = True
        mock_accessibility.return_value = False
        
        result = check_all_permissions()
        
        expected = {
            "screen_recording": True,
            "accessibility": False
        }
        assert result == expected
    
    @patch('motorways.utils.permissions.check_screen_recording_permission')
    @patch('motorways.utils.permissions.check_accessibility_permission')
    def test_check_all_permissions_none_granted(self, mock_accessibility, mock_screen_recording):
        """Test checking all permissions when none are granted."""
        mock_screen_recording.return_value = False
        mock_accessibility.return_value = False
        
        result = check_all_permissions()
        
        expected = {
            "screen_recording": False,
            "accessibility": False
        }
        assert result == expected
    
    def test_get_permission_instructions(self):
        """Test getting permission instructions."""
        instructions = get_permission_instructions()
        
        assert "screen_recording" in instructions
        assert "accessibility" in instructions
        
        # Check that instructions contain key information
        screen_instr = instructions["screen_recording"]
        assert "Screen Recording" in screen_instr
        assert "System Preferences" in screen_instr or "Settings" in screen_instr
        
        access_instr = instructions["accessibility"]
        assert "Accessibility" in access_instr
        assert "System Preferences" in access_instr or "Settings" in access_instr
    
    @patch('motorways.utils.permissions.check_all_permissions')
    def test_validate_permissions_or_exit_success(self, mock_check_permissions):
        """Test validate_permissions_or_exit when all permissions granted."""
        mock_check_permissions.return_value = {
            "screen_recording": True,
            "accessibility": True
        }
        
        # Should not raise SystemExit
        from motorways.utils.permissions import validate_permissions_or_exit
        validate_permissions_or_exit()  # Should complete without exception
    
    @patch('motorways.utils.permissions.check_all_permissions')
    def test_validate_permissions_or_exit_failure(self, mock_check_permissions):
        """Test validate_permissions_or_exit when permissions missing."""
        mock_check_permissions.return_value = {
            "screen_recording": False,
            "accessibility": True
        }
        
        from motorways.utils.permissions import validate_permissions_or_exit
        
        with pytest.raises(SystemExit) as exc_info:
            validate_permissions_or_exit()
        
        assert exc_info.value.code == 1


class TestPermissionsIntegration:
    """Integration-style tests for permissions."""
    
    @patch('motorways.utils.permissions.check_all_permissions')
    @patch('builtins.input')
    def test_prompt_for_permission_grant_success(self, mock_input, mock_check_permissions):
        """Test prompting for permission grant when user confirms."""
        # First call returns missing permissions, second call returns granted
        mock_check_permissions.side_effect = [
            {"screen_recording": False, "accessibility": True},  # Initial check
            {"screen_recording": True, "accessibility": True}    # After user action
        ]
        mock_input.return_value = "y"
        
        from motorways.utils.permissions import prompt_for_permission_grant
        
        result = prompt_for_permission_grant()
        
        assert result is True
        assert mock_check_permissions.call_count == 2
    
    @patch('motorways.utils.permissions.check_all_permissions')
    @patch('builtins.input')
    def test_prompt_for_permission_grant_still_missing(self, mock_input, mock_check_permissions):
        """Test prompting for permission grant when permissions still missing."""
        # Both calls return missing permissions
        mock_check_permissions.return_value = {
            "screen_recording": False, 
            "accessibility": True
        }
        mock_input.return_value = "y"
        
        from motorways.utils.permissions import prompt_for_permission_grant
        
        result = prompt_for_permission_grant()
        
        assert result is False
    
    @patch('motorways.utils.permissions.check_all_permissions')
    @patch('builtins.input')
    def test_prompt_for_permission_grant_user_declines(self, mock_input, mock_check_permissions):
        """Test prompting for permission grant when user declines."""
        mock_check_permissions.return_value = {
            "screen_recording": False,
            "accessibility": True
        }
        mock_input.return_value = "n"
        
        from motorways.utils.permissions import prompt_for_permission_grant
        
        result = prompt_for_permission_grant()
        
        assert result is False
        # Should only call check_permissions once (initial check)
        assert mock_check_permissions.call_count == 1
    
    @patch('motorways.utils.permissions.check_all_permissions')
    def test_prompt_for_permission_grant_already_granted(self, mock_check_permissions):
        """Test prompting when all permissions already granted."""
        mock_check_permissions.return_value = {
            "screen_recording": True,
            "accessibility": True
        }
        
        from motorways.utils.permissions import prompt_for_permission_grant
        
        result = prompt_for_permission_grant()
        
        assert result is True
        # Should only call check_permissions once
        assert mock_check_permissions.call_count == 1