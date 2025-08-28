# Gunicorn configuration file

# Worker class for handling asynchronous requests
worker_class = 'gevent'

# Number of worker processes
workers = 4

# The socket to bind to
bind = '0.0.0.0:8080'

# Worker timeout for handling long-running requests (in seconds)
timeout = 120

# Keep-alive connections
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
