import logging
try:
    import facenet
except ImportError:
    logging.warning("Could not import facenet")