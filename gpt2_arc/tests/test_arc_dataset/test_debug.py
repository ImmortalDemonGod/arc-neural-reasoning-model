# src/data/test_arc_dataset_debug.py

import pytest
import logging
from src.data.arc_dataset import ARCDataset
from src.data.utils.custom_exceptions import DataLoadingError

# Enable debug logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_data():
    """Create minimal test data"""
    return [
        {
            "task_id": "test1",
            "input": [[[1, 0], [0, 1]]],
            "output": [[[0, 1], [1, 0]]]
        }
    ]

def test_basic_loading(sample_data, caplog):
    """Test basic data loading with debug output"""
    caplog.set_level(logging.DEBUG)
    
    logger.debug("Starting basic loading test")
    dataset = ARCDataset(data_source=sample_data)
    
    # Print all debug messages
    for record in caplog.records:
        print(f"LOG: {record.getMessage()}")
    
    assert len(dataset) > 0

def test_missing_fields(caplog):
    """Test handling of missing fields with debug output"""
    caplog.set_level(logging.DEBUG)
    
    incomplete_data = [
        {"input": [[[1, 2], [3, 4]]]}  # Missing output
    ]
    
    logger.debug("Testing with incomplete data")
    logger.debug(f"Test data: {incomplete_data}")
    
    # Add step-by-step debugging
    try:
        logger.debug("Step 1: Creating dataset")
        dataset = ARCDataset(data_source=incomplete_data)
        
        logger.debug("Step 2: Dataset created successfully")
        logger.debug(f"Dataset length: {len(dataset)}")
        logger.debug(f"Dataset attributes: {vars(dataset)}")
        
        logger.debug("Step 3: Attempting to access first item")
        item = dataset[0]
        
        logger.debug("Step 4: Successfully accessed item")
        logger.debug(f"Retrieved item: {item}")
        
        raise AssertionError("Should have raised DataLoadingError")
    except Exception as e:
        logger.debug(f"Caught exception: {type(e).__name__} - {str(e)}")
        logger.debug(f"Exception details: {vars(e)}")
        
        # Print all captured logs
        print("\nFull debug log:")
        for record in caplog.records:
            print(f"LOG [{record.levelname}]: {record.getMessage()}")
            
        if not isinstance(e, DataLoadingError):
            logger.error(f"Expected DataLoadingError, but got {type(e).__name__}")
            logger.error("Stack trace:", exc_info=True)
            
        assert isinstance(e, DataLoadingError)