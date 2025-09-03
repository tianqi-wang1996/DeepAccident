"""Validation tests to ensure the testing infrastructure is working correctly."""

import pytest
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


class TestInfrastructureValidation:
    """Test class to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_pytest_is_working(self):
        """Basic test to ensure pytest is functioning."""
        assert True
    
    @pytest.mark.unit
    def test_fixtures_are_available(self, temp_dir, mock_config):
        """Test that shared fixtures are properly loaded."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert mock_config is not None
    
    @pytest.mark.unit
    def test_sample_data_fixtures(self, sample_point_cloud, sample_bbox_3d):
        """Test that sample data fixtures work correctly."""
        assert isinstance(sample_point_cloud, torch.Tensor)
        assert sample_point_cloud.shape[1] == 4  # x, y, z, intensity
        assert sample_point_cloud.dtype == torch.float32
        
        assert 'boxes' in sample_bbox_3d
        assert 'labels' in sample_bbox_3d
        assert 'scores' in sample_bbox_3d
        assert sample_bbox_3d['boxes'].shape[1] == 7  # 3D bbox format
    
    @pytest.mark.unit
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_available(self):
        """Test that PyTorch is available."""
        x = torch.tensor([1, 2, 3])
        assert x.sum().item() == 6
    
    @pytest.mark.unit
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_numpy_available(self):
        """Test that NumPy is available."""
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
    
    @pytest.mark.unit
    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temporary directory fixture creates and will be cleaned up."""
        test_file = temp_dir / "test_file.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        # Directory cleanup happens automatically after test
    
    @pytest.mark.integration
    def test_mock_components_integration(self, mock_model, mock_optimizer, mock_dataset):
        """Integration test for mock components working together."""
        # Simulate a training step
        mock_optimizer.zero_grad()
        loss = mock_model.forward()
        mock_optimizer.step()
        
        assert 'loss' in loss
        assert mock_dataset.__len__() == 100
        assert len(mock_dataset.CLASSES) == 3
    
    @pytest.mark.slow
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_device_detection(self, device):
        """Test device detection (marked as slow for demonstration)."""
        assert device in [torch.device('cpu'), torch.device('cuda')]
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not NUMPY_AVAILABLE, reason="PyTorch or NumPy not available")
    def test_voxel_data_structure(self, sample_voxels):
        """Test voxel data structure fixture."""
        assert 'voxels' in sample_voxels
        assert 'num_points' in sample_voxels
        assert 'coordinates' in sample_voxels
        
        voxels = sample_voxels['voxels']
        assert voxels.dim() == 3  # (num_voxels, max_points, features)
        assert voxels.shape[2] == 4  # x, y, z, intensity
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not NUMPY_AVAILABLE, reason="PyTorch or NumPy not available")
    def test_reproducible_state(self, reproducible_random_state):
        """Test that random state is reproducible."""
        # Generate some random numbers
        np_rand1 = np.random.random(5)
        torch_rand1 = torch.rand(5)
        
        # Reset the fixture and generate again
        np.random.seed(42)
        torch.manual_seed(42)
        
        np_rand2 = np.random.random(5)
        torch_rand2 = torch.rand(5)
        
        # Should be the same due to fixed seed
        np.testing.assert_array_equal(np_rand1, np_rand2)
        torch.testing.assert_allclose(torch_rand1, torch_rand2)


class TestModuleImports:
    """Test that key modules can be imported."""
    
    @pytest.mark.unit
    def test_mmdet3d_import(self):
        """Test that mmdet3d package can be imported."""
        try:
            import mmdet3d
            assert hasattr(mmdet3d, '__version__')
        except ImportError:
            pytest.skip("mmdet3d not installed, skipping import test")
    
    @pytest.mark.unit
    @pytest.mark.skipif(not TORCH_AVAILABLE or not NUMPY_AVAILABLE, reason="PyTorch or NumPy not available")
    def test_core_dependencies(self):
        """Test that core dependencies are available."""
        import torch
        import numpy as np
        
        assert torch.__version__ is not None
        assert np.__version__ is not None


class TestProjectStructure:
    """Test that the project structure is as expected."""
    
    @pytest.mark.unit
    def test_project_root_exists(self):
        """Test that we're in the right project root."""
        project_root = Path.cwd()
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "mmdet3d").exists()
        assert (project_root / "tests").exists()
    
    @pytest.mark.unit
    def test_test_directory_structure(self):
        """Test that test directory structure is correct."""
        tests_dir = Path.cwd() / "tests"
        assert tests_dir.exists()
        assert (tests_dir / "__init__.py").exists()
        assert (tests_dir / "conftest.py").exists()
        assert (tests_dir / "unit").exists()
        assert (tests_dir / "integration").exists()
        assert (tests_dir / "unit" / "__init__.py").exists()
        assert (tests_dir / "integration" / "__init__.py").exists()