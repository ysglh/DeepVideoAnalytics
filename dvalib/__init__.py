import logging
try:
    import entity,facenet
except ImportError:
    logging.warning("Could not import entity,facenet")