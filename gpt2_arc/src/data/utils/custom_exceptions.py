# gpt2_arc/src/data/utils/custom_exceptions.py
class ARCDatasetError(Exception):
    """Base exception for ARCDataset"""
    pass

class DataLoadingError(ARCDatasetError):
    """Raised when data loading fails"""
    pass

class ValidationError(ARCDatasetError):
    """Raised when data validation fails"""
    pass

class ResourceError(ARCDatasetError):
    """Raised when resource management fails"""
    pass