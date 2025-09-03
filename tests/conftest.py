"""Shared pytest fixtures for MMDetection3D testing."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = Mock()
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = Mock()
    NUMPY_AVAILABLE = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = Mock()
    config.data = Mock()
    config.model = Mock()
    config.train_cfg = Mock()
    config.test_cfg = Mock()
    config.optimizer = Mock()
    config.lr_config = Mock()
    config.checkpoint_config = Mock()
    config.log_config = Mock()
    return config


@pytest.fixture
def sample_point_cloud():
    """Generate a sample point cloud for testing."""
    if not NUMPY_AVAILABLE or not TORCH_AVAILABLE:
        pytest.skip("NumPy or PyTorch not available")
    num_points = 1000
    points = np.random.rand(num_points, 4).astype(np.float32)  # x, y, z, intensity
    return torch.from_numpy(points)


@pytest.fixture
def sample_bbox_3d():
    """Generate sample 3D bounding boxes for testing."""
    if not NUMPY_AVAILABLE or not TORCH_AVAILABLE:
        pytest.skip("NumPy or PyTorch not available")
    num_boxes = 5
    boxes = np.random.rand(num_boxes, 7).astype(np.float32)  # x, y, z, w, l, h, r
    labels = np.random.randint(0, 3, num_boxes)  # 3 classes
    scores = np.random.rand(num_boxes).astype(np.float32)
    return {
        'boxes': torch.from_numpy(boxes),
        'labels': torch.from_numpy(labels),
        'scores': torch.from_numpy(scores)
    }


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    dataset.CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    dataset.cat2id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
    dataset.flag = np.zeros(100, dtype=np.uint8)
    return dataset


@pytest.fixture
def mock_dataloader():
    """Mock dataloader for testing."""
    dataloader = Mock()
    dataloader.dataset = Mock()
    dataloader.dataset.CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    return dataloader


@pytest.fixture
def sample_data_info():
    """Sample data info dictionary for testing."""
    return {
        'sample_idx': 0,
        'pts_filename': 'samples/LIDAR_TOP/test.bin',
        'img_filename': 'samples/CAM_FRONT/test.jpg',
        'lidar2img': np.eye(4).astype(np.float32),
        'ann_info': {
            'gt_bboxes_3d': np.random.rand(3, 7).astype(np.float32),
            'gt_names': ['Car', 'Car', 'Pedestrian'],
            'gt_labels_3d': np.array([0, 0, 1])
        }
    }


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = MagicMock()
    model.train = Mock(return_value=None)
    model.eval = Mock(return_value=None)
    model.forward = Mock(return_value={'loss': torch.tensor(1.0)})
    model.forward_test = Mock(return_value=[{'boxes_3d': torch.rand(5, 7)}])
    return model


@pytest.fixture
def mock_optimizer():
    """Mock optimizer for testing."""
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    optimizer.param_groups = [{'lr': 0.001}]
    return optimizer


@pytest.fixture
def sample_voxels():
    """Generate sample voxel data for testing."""
    max_points = 32
    max_voxels = 20000
    voxel_size = [0.16, 0.16, 4]
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    
    num_voxels = np.random.randint(100, 1000)
    voxels = np.random.rand(num_voxels, max_points, 4).astype(np.float32)
    num_points = np.random.randint(1, max_points + 1, num_voxels)
    coordinates = np.random.randint(0, 100, (num_voxels, 3))
    
    return {
        'voxels': torch.from_numpy(voxels),
        'num_points': torch.from_numpy(num_points),
        'coordinates': torch.from_numpy(coordinates)
    }


@pytest.fixture
def setup_test_environment(tmp_path, monkeypatch):
    """Set up a complete test environment with temporary directories."""
    work_dir = tmp_path / "work_dir"
    work_dir.mkdir()
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    monkeypatch.setenv("WORK_DIR", str(work_dir))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    
    return {
        'work_dir': work_dir,
        'data_dir': data_dir,
        'tmp_path': tmp_path
    }


@pytest.fixture(scope="session")
def test_data_root():
    """Root directory for test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture
def disable_gpu(monkeypatch):
    """Force CPU usage for testing."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch, 'cuda', Mock())


@pytest.fixture
def reproducible_random_state():
    """Set reproducible random state for testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)