This project implements a basic **Decentralized P2P File Sharing and Synchronization System** in a single Python file. Each peer acts as both a server (listening for incoming connections) and a client (connecting to other peers). The system allows peers to discover files shared by others, download missing files, and keep local files synchronized with the latest versions found across the network.

**Key Features:**
-   **Peer-to-Peer Communication:** Peers connect directly using TCP sockets to exchange file metadata and transfer files.
-   **File Hashing:** SHA256 hashes are used to verify file integrity and detect changes.
-   **Synchronization Logic:** Peers periodically scan their local shared directory and compare their file lists with known peers. If a file is missing or an older version is detected, the peer requests the newer version. "Last modified wins" is the basic conflict resolution strategy.
-   **Decentralized:** No central server is required for file storage or coordination.
-   **Real-time Monitoring (Optional):** Integrates with the `watchdog` library to detect local file system changes instantly, triggering faster updates. Falls back to periodic scanning if `watchdog` is not installed.
-   **Command-Line Interface:** Configurable through command-line arguments for host, port, shared directory, and initial known peers.
-   **Graceful Shutdown:** Handles `Ctrl+C` and explicit `exit` command to clean up resources.
-   **Logging:** Provides detailed logs for operation monitoring and debugging.

**Limitations (for a production-grade system):**
-   **Security:** Lacks encryption (e.g., TLS/SSL) for communication and authentication mechanisms, making it vulnerable to eavesdropping and malicious peers.
-   **NAT Traversal:** Assumes direct network reachability between peers (e.g., on a local network or with public IPs/port forwarding). It does not implement NAT traversal techniques.
-   **Advanced Conflict Resolution:** Uses a simple "last modified wins" strategy. More complex systems would include versioning, merging, or user intervention for conflicts.
-   **Peer Discovery:** Relies on a hardcoded list of initial known peers. A more robust system would use a Distributed Hash Table (DHT) or local network broadcast for dynamic discovery.
-   **Scalability:** Uses `threading` for concurrent connections, which scales reasonably well for a small number of peers but may have limitations for very large networks.
-   **File Transfer Features:** Lacks features like file transfer resume, block-level synchronization (like `rsync`), or simultaneous downloads from multiple peers.

To run this code:
1.  Save it as a Python file (e.g., `p2p_sync.py`).
2.  (Optional, but recommended): Install `watchdog` for real-time file monitoring: `pip install watchdog`.
3.  Run multiple instances from different terminals, specifying unique ports and shared directories.
    *   **Peer 1 (e.g., running on port 9000, sharing 'shared_files_9000'):**
        `python p2p_sync.py --host 127.0.0.1 --port 9000`
    *   **Peer 2 (e.g., running on port 9001, sharing 'shared_files_9001', connected to Peer 1):**
        `python p2p_sync.py --host 127.0.0.1 --port 9001 --peers 127.0.0.1:9000`
    *   **Peer 3 (e.g., running on port 9002, sharing 'shared_files_9002', connected to Peer 1 and Peer 2):**
        `python p2p_sync.py --host 127.0.0.1 --port 9002 --peers 127.0.0.1:9000 127.0.0.1:9001`

    You can then place files into one peer's `shared_files_<port>` directory, and after a short while (or immediate if watchdog is enabled), they should propagate to other connected peers.

```python
import os
import sys
import socket
import json
import hashlib
import threading
import time
import logging
import argparse
from datetime import datetime

# --- Optional: Watchdog for real-time file system monitoring ---
# If you don't want to install watchdog, comment out the following two lines
# and the `FileChangeHandler` class, then the `_start_file_monitor` method.
# The system will then rely solely on periodic scanning for changes.
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Watchdog not found. Real-time file monitoring will be disabled. "
          "Install with 'pip install watchdog' for better synchronization.", file=sys.stderr)
# -----------------------------------------------------------------

# --- Configuration Constants ---
BUFFER_SIZE = 4096  # Standard buffer size for socket operations (in bytes)
PROTOCOL_VERSION = "1.0"
DEFAULT_PORT = 9000
DEFAULT_SHARED_DIR = "shared_files"
SYNC_INTERVAL_SECONDS = 30  # How often to initiate sync with known peers (if no watchdog or for initial sync)
PEER_TIMEOUT = 5  # Seconds to wait for a peer response during network operations

# --- Logging Setup ---
# Configure logging to show timestamps, thread names, log level, and message
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def calculate_file_hash(filepath):
    """
    Calculates the SHA256 hash of a file.

    Args:
        filepath (str): The full path to the file.

    Returns:
        str or None: The hexadecimal SHA256 hash of the file, or None if an error occurs.
    """
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(BUFFER_SIZE):
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        logger.error(f"Error reading file {filepath} for hashing: {e}")
        return None

def send_json_message(sock, message_type, payload):
    """
    Sends a JSON-encoded message over the socket.
    The message is prefixed with its length to ensure the receiver knows how much data to expect.

    Args:
        sock (socket.socket): The socket to send the message through.
        message_type (str): The type of message (e.g., 'HELLO', 'FILE_LIST_REQUEST').
        payload (dict): The data payload of the message.

    Returns:
        bool: True if the message was sent successfully, False otherwise.
    """
    message = {
        "version": PROTOCOL_VERSION,
        "type": message_type,
        "payload": payload
    }
    encoded_message = json.dumps(message).encode('utf-8')
    # Prepend message length (4 bytes, big-endian)
    length_prefix = len(encoded_message).to_bytes(4, 'big')
    try:
        sock.sendall(length_prefix + encoded_message)
        return True
    except socket.error as e:
        logger.error(f"Socket error sending message: {e}")
        return False
    except Exception as e:
        logger.error(f"Error preparing or sending JSON message: {e}")
        return False

def receive_json_message(sock, timeout=PEER_TIMEOUT):
    """
    Receives a JSON-encoded message from the socket.
    It first reads a 4-byte length prefix, then the full message.

    Args:
        sock (socket.socket): The socket to receive the message from.
        timeout (int): The maximum time to wait for data (in seconds).

    Returns:
        dict or None: The decoded JSON message as a dictionary, or None if an error occurs
                      or the connection is closed.
    """
    sock.settimeout(timeout)
    try:
        # First, receive the 4-byte length prefix
        length_bytes = sock.recv(4)
        if not length_bytes:
            # Connection closed by peer or timed out on empty read
            return None
        if len(length_bytes) < 4:
            logger.warning("Received incomplete length prefix.")
            return None
        message_length = int.from_bytes(length_bytes, 'big')

        # Receive the actual message data
        chunks = []
        bytes_recd = 0
        while bytes_recd < message_length:
            chunk = sock.recv(min(message_length - bytes_recd, BUFFER_SIZE))
            if not chunk:
                # Connection closed prematurely
                break
            chunks.append(chunk)
            bytes_recd += len(chunk)
        
        if bytes_recd < message_length:
            logger.warning(f"Incomplete message received. Expected {message_length}, got {bytes_recd} bytes.")
            return None

        full_message = b"".join(chunks).decode('utf-8')
        return json.loads(full_message)
    except socket.timeout:
        # logger.debug("Socket receive timed out.") # Often expected for server accept() or client waits
        return None
    except (socket.error, json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error receiving or decoding message: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in receive_json_message: {e}")
        return None
    finally:
        sock.settimeout(None) # Reset timeout to blocking or default

# --- Peer Class ---

class Peer:
    """
    Represents a single peer in the decentralized file sharing and synchronization network.
    Each peer manages its own shared directory, listens for incoming connections,
    and connects to other known peers to synchronize files.
    """
    def __init__(self, host, port, shared_dir, known_peers=None):
        """
        Initializes the Peer instance.

        Args:
            host (str): The IP address this peer will bind to.
            port (int): The port number this peer will listen on.
            shared_dir (str): The path to the directory to be shared and synchronized.
            known_peers (list): An optional list of (host, port) tuples for initial known peers.
        """
        self.host = host
        self.port = port
        self.shared_dir = os.path.abspath(shared_dir)
        os.makedirs(self.shared_dir, exist_ok=True) # Ensure the shared directory exists
        
        # Set of (host, port) tuples of all known peers, including self but excluded in sync logic
        self.peers = set(known_peers) if known_peers else set() 
        # Add self to known peers to make sure it's part of the network model, though not connected
        self.peers.add((self.host, self.port)) 

        self.active_peer_connections = {} # { (host, port): socket_object } for active outgoing connections
        self.peer_file_states = {} # { (host, port): { filepath_relative: metadata } } from remote peers

        # Local file metadata: { relative_path: { 'hash': ..., 'timestamp': ..., 'size': ... } }
        self.local_file_metadata = {} 
        self.server_socket = None # The socket for listening to incoming connections
        self.running = True     # Flag to control the main loops of threads
        
        self.sync_lock = threading.Lock() # To prevent race conditions during file sync operations
        self.threads = [] # List to keep track of all spawned threads for potential management

        logger.info(f"Peer initialized: {self.host}:{self.port}, Shared Dir: {self.shared_dir}")
        logger.info(f"Initial known peers: {self.peers}")

    def _get_local_file_metadata(self, relative_path):
        """
        Generates metadata for a single local file.

        Args:
            relative_path (str): The path of the file relative to the shared directory.

        Returns:
            dict or None: A dictionary containing 'path', 'hash', 'timestamp', and 'size'
                          if the file exists and is readable, None otherwise.
        """
        filepath = os.path.join(self.shared_dir, relative_path)
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            return None
        
        file_hash = calculate_file_hash(filepath)
        if file_hash is None:
            return None # Hashing failed
        
        timestamp = os.path.getmtime(filepath)
        size = os.path.getsize(filepath)
        return {
            'path': relative_path,
            'hash': file_hash,
            'timestamp': timestamp,
            'size': size
        }

    def _scan_local_files(self):
        """
        Scans the shared directory for files and updates the `local_file_metadata` dictionary.
        Optimized to avoid re-hashing unchanged files by checking modification timestamps.
        """
        current_metadata = {}
        for root, _, files in os.walk(self.shared_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, self.shared_dir)
                
                # Check if file has changed since last scan or is new
                old_meta = self.local_file_metadata.get(relative_path)
                
                try:
                    current_timestamp = os.path.getmtime(full_path)
                except FileNotFoundError:
                    # File might have been deleted between os.walk and getmtime
                    continue

                if old_meta and old_meta['timestamp'] == current_timestamp:
                    # File hasn't been modified, reuse old metadata
                    current_metadata[relative_path] = old_meta
                else:
                    # File is new or has been modified, regenerate metadata
                    meta = self._get_local_file_metadata(relative_path)
                    if meta:
                        current_metadata[relative_path] = meta
        
        self.local_file_metadata = current_metadata
        # logger.debug(f"Local files scanned. Total: {len(self.local_file_metadata)} files.")

    def _handle_incoming_connection(self, client_sock, client_addr):
        """
        Handles a new incoming connection from another peer.
        Performs a handshake, then processes incoming messages in a loop.

        Args:
            client_sock (socket.socket): The socket for the client connection.
            client_addr (tuple): The (host, port) address of the connected client.
        """
        logger.info(f"Incoming connection from {client_addr}")
        peer_id = None # To store (host, port) of the connected peer
        try:
            # Step 1: Receive HELLO message
            message = receive_json_message(client_sock)
            if not message or message.get('type') != 'HELLO':
                logger.warning(f"Invalid or missing HELLO message from {client_addr}. Closing connection.")
                return

            peer_host = message['payload']['host']
            peer_port = message['payload']['port']
            peer_id = (peer_host, peer_port)
            self.peers.add(peer_id) # Add new peer to known list

            # Step 2: Send HELLO_ACK
            if not send_json_message(client_sock, 'HELLO_ACK', {'host': self.host, 'port': self.port}):
                logger.warning(f"Failed to send HELLO_ACK to {peer_id}. Closing connection.")
                return
            logger.info(f"Handshake complete with {peer_id}")

            # Main message loop for this connection
            while self.running:
                message = receive_json_message(client_sock)
                if not message:
                    logger.info(f"Peer {peer_id} disconnected or timeout.")
                    break

                msg_type = message.get('type')
                payload = message.get('payload')

                if msg_type == 'FILE_LIST_REQUEST':
                    logger.debug(f"Received FILE_LIST_REQUEST from {peer_id}")
                    self._scan_local_files() # Ensure we have the latest local data before sending
                    send_json_message(client_sock, 'FILE_LIST_RESPONSE', self.local_file_metadata)
                
                elif msg_type == 'FILE_REQUEST':
                    file_path_relative = payload.get('path')
                    logger.info(f"Received FILE_REQUEST for '{file_path_relative}' from {peer_id}")
                    self._send_file(client_sock, file_path_relative)
                
                else:
                    logger.warning(f"Unknown message type '{msg_type}' from {peer_id}")

        except Exception as e:
            logger.error(f"Error handling connection from {client_addr} (peer {peer_id}): {e}")
        finally:
            client_sock.close()
            logger.info(f"Connection with {client_addr} (peer {peer_id}) closed.")

    def _start_server(self):
        """
        Starts the peer's listening server for incoming connections.
        Accepts connections and spawns new threads to handle each client.
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5) # Max 5 queued connections
            logger.info(f"Peer listening on {self.host}:{self.port}")
            self._scan_local_files() # Initial scan on startup
        except socket.error as e:
            logger.critical(f"Failed to start server on {self.host}:{self.port}: {e}")
            self.running = False # Signal other threads to stop as well
            return

        while self.running:
            try:
                self.server_socket.settimeout(1.0) # Allow server to periodically check self.running flag
                client_sock, client_addr = self.server_socket.accept()
                thread = threading.Thread(target=self._handle_incoming_connection, args=(client_sock, client_addr), daemon=True)
                thread.name = f"Handler-{client_addr[0]}:{client_addr[1]}"
                self.threads.append(thread)
                thread.start()
            except socket.timeout:
                pass # Expected behavior to check self.running periodically
            except socket.error as e:
                if self.running: # Only log if it's an unexpected error, not due to shutdown
                    logger.error(f"Server socket error: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error in server loop: {e}")
                break
        
        logger.info(f"Server on {self.host}:{self.port} stopped.")

    def _connect_to_peer(self, peer_host, peer_port):
        """
        Establishes an outgoing connection to another peer and performs a handshake.

        Args:
            peer_host (str): The host IP address of the peer to connect to.
            peer_port (int): The port number of the peer to connect to.

        Returns:
            socket.socket or None: The established socket object if successful, None otherwise.
        """
        if (peer_host, peer_port) == (self.host, self.port):
            return None # Prevent connecting to self

        # Reuse existing connection if active
        if (peer_host, peer_port) in self.active_peer_connections:
            # Check if socket is still active (e.g., by a quick non-blocking peek)
            # This is a basic check. A more robust one might use PING/PONG.
            try:
                # Attempt a non-blocking recv to see if connection is alive.
                # If it raises an error (e.g., connection reset), it's dead.
                # If it receives data, it's alive. If it raises timeout/wouldblock, it's alive.
                self.active_peer_connections[(peer_host, peer_port)].settimeout(0.1)
                data = self.active_peer_connections[(peer_host, peer_port)].recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                self.active_peer_connections[(peer_host, peer_port)].settimeout(None) # Reset
                if len(data) == 0: # Peer closed connection
                    raise socket.error("Peer closed connection unexpectedly.")
                return self.active_peer_connections[(peer_host, peer_port)]
            except (socket.timeout, BlockingIOError):
                # Socket is still alive and no data waiting, good.
                self.active_peer_connections[(peer_host, peer_port)].settimeout(None) # Reset
                return self.active_peer_connections[(peer_host, peer_port)]
            except socket.error as e:
                logger.warning(f"Existing connection to {peer_host}:{peer_port} found dead: {e}. Reconnecting.")
                self.active_peer_connections[(peer_host, peer_port)].close()
                del self.active_peer_connections[(peer_host, peer_port)]

        logger.info(f"Attempting to connect to peer {peer_host}:{peer_port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(PEER_TIMEOUT)
            sock.connect((peer_host, peer_port))
            
            # Step 1: Send HELLO message
            if not send_json_message(sock, 'HELLO', {'host': self.host, 'port': self.port}):
                sock.close()
                return None
            
            # Step 2: Receive HELLO_ACK
            response = receive_json_message(sock)
            if not response or response.get('type') != 'HELLO_ACK':
                logger.warning(f"Invalid or missing HELLO_ACK from {peer_host}:{peer_port}. Closing connection.")
                sock.close()
                return None
            
            self.active_peer_connections[(peer_host, peer_port)] = sock
            logger.info(f"Successfully connected to {peer_host}:{peer_port}")
            sock.settimeout(None) # Reset timeout for connected sockets to blocking
            return sock
        except socket.error as e:
            logger.warning(f"Could not connect to peer {peer_host}:{peer_port}: {e}")
            sock.close()
            return None
        except Exception as e:
            logger.error(f"Error during connection to {peer_host}:{peer_port}: {e}")
            sock.close()
            return None
        finally:
            # Ensure timeout is reset on the socket before returning or closing
            sock.settimeout(None)

    def _send_file(self, sock, relative_path):
        """
        Sends a requested file to a connected peer.

        Args:
            sock (socket.socket): The socket to send the file through.
            relative_path (str): The path of the file relative to the shared directory.

        Returns:
            bool: True if the file was sent successfully, False otherwise.
        """
        full_path = os.path.join(self.shared_dir, relative_path)
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            logger.warning(f"File '{relative_path}' not found locally for transfer to peer. Path: {full_path}")
            send_json_message(sock, 'FILE_NOT_FOUND', {'path': relative_path})
            return False

        try:
            with open(full_path, 'rb') as f:
                file_metadata = self._get_local_file_metadata(relative_path)
                if not file_metadata:
                    raise IOError("Could not get metadata for file before sending.")

                # Send file metadata first
                if not send_json_message(sock, 'FILE_DATA', {
                    'path': relative_path,
                    'hash': file_metadata['hash'],
                    'timestamp': file_metadata['timestamp'],
                    'size': file_metadata['size']
                }):
                    return False # Failed to send metadata

                # Send file content
                bytes_sent = 0
                while chunk := f.read(BUFFER_SIZE):
                    sock.sendall(chunk)
                    bytes_sent += len(chunk)
                logger.info(f"Sent {bytes_sent} bytes for file '{relative_path}' to peer.")
            return True
        except socket.error as e:
            logger.error(f"Socket error sending file '{relative_path}': {e}")
            return False
        except IOError as e:
            logger.error(f"File I/O error sending '{relative_path}': {e}")
            send_json_message(sock, 'FILE_NOT_FOUND', {'path': relative_path}) # Notify peer if possible
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending file '{relative_path}': {e}")
            return False

    def _receive_file(self, sock, file_info):
        """
        Receives a file from a connected peer based on the provided file information.
        Verifies the received file's hash and sets its timestamp.

        Args:
            sock (socket.socket): The socket to receive the file from.
            file_info (dict): A dictionary containing 'path', 'hash', 'timestamp', and 'size' of the file.

        Returns:
            bool: True if the file was received and verified successfully, False otherwise.
        """
        relative_path = file_info['path']
        expected_hash = file_info['hash']
        expected_timestamp = file_info['timestamp']
        expected_size = file_info['size']
        
        full_path = os.path.join(self.shared_dir, relative_path)
        # Ensure the directory structure exists for the incoming file
        os.makedirs(os.path.dirname(full_path), exist_ok=True) 

        logger.info(f"Receiving file '{relative_path}' ({expected_size} bytes)...")
        received_bytes = 0
        hasher = hashlib.sha256()

        try:
            with open(full_path, 'wb') as f:
                # Use a specific timeout for receiving file data to prevent indefinite blocking
                sock.settimeout(PEER_TIMEOUT * 2) # Give more time for large files
                while received_bytes < expected_size:
                    chunk = sock.recv(min(expected_size - received_bytes, BUFFER_SIZE))
                    if not chunk:
                        logger.error(f"Connection closed prematurely while receiving '{relative_path}'. Received {received_bytes}/{expected_size} bytes.")
                        return False # Indicate failure
                    f.write(chunk)
                    hasher.update(chunk)
                    received_bytes += len(chunk)
            
            # Verify hash after receiving the entire file
            actual_hash = hasher.hexdigest()
            if actual_hash != expected_hash:
                logger.error(f"Hash mismatch for '{relative_path}'. Expected {expected_hash}, got {actual_hash}. Deleting corrupt file.")
                os.remove(full_path)
                return False
            
            # Set file modification timestamp to match the remote peer's timestamp
            os.utime(full_path, (time.time(), expected_timestamp)) # Access time, Mod time

            logger.info(f"Successfully received and verified '{relative_path}'. Size: {received_bytes} bytes.")
            return True
        except socket.timeout:
            logger.error(f"Socket timed out while receiving '{relative_path}'.")
        except socket.error as e:
            logger.error(f"Socket error receiving file '{relative_path}': {e}")
        except IOError as e:
            logger.error(f"File I/O error receiving '{relative_path}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error receiving file '{relative_path}': {e}")
        finally:
            sock.settimeout(None) # Reset socket timeout
            # Cleanup incomplete/corrupt file if it exists and transfer failed
            if not os.path.exists(full_path) or calculate_file_hash(full_path) != expected_hash:
                if os.path.exists(full_path):
                    logger.warning(f"Cleaning up partially received/corrupt file: {full_path}")
                    os.remove(full_path)
        return False

    def _synchronize_with_peer(self, peer_id):
        """
        Initiates a synchronization round with a specific connected peer.
        This peer (client) requests the remote peer's file list, compares it with its own,
        and requests any missing or newer files.

        Args:
            peer_id (tuple): The (host, port) of the peer to synchronize with.
        """
        peer_host, peer_port = peer_id
        
        with self.sync_lock: # Ensure only one sync operation runs at a time for this peer
            logger.info(f"Starting synchronization with {peer_id}...")
            
            # Establish or reuse connection
            sock = self._connect_to_peer(peer_host, peer_port)
            if not sock:
                logger.warning(f"Could not establish or reuse connection to {peer_id} for sync.")
                return

            try:
                # 1. Request peer's file list
                if not send_json_message(sock, 'FILE_LIST_REQUEST', {}):
                    logger.warning(f"Failed to send FILE_LIST_REQUEST to {peer_id}.")
                    return

                peer_files_response = receive_json_message(sock)

                if not peer_files_response or peer_files_response.get('type') != 'FILE_LIST_RESPONSE':
                    logger.warning(f"Failed to get file list from {peer_id}. Response: {peer_files_response}")
                    return
                
                remote_file_metadata = peer_files_response['payload']
                self.peer_file_states[peer_id] = remote_file_metadata
                logger.debug(f"Received {len(remote_file_metadata)} files from {peer_id}.")

                self._scan_local_files() # Ensure local metadata is fresh for comparison

                # 2. Compare local files with peer's files
                files_to_download = []

                # Check for files missing locally or newer on peer
                for remote_path, remote_meta in remote_file_metadata.items():
                    local_meta = self.local_file_metadata.get(remote_path)

                    if not local_meta:
                        # File exists on peer, not locally. Schedule for download.
                        files_to_download.append(remote_meta)
                        logger.debug(f"File '{remote_path}' exists on peer, not locally. Will download.")
                    elif local_meta['hash'] != remote_meta['hash']:
                        # File exists on both, but content differs. Compare timestamps.
                        if remote_meta['timestamp'] > local_meta['timestamp']:
                            # Peer has a newer version. Schedule for download.
                            files_to_download.append(remote_meta)
                            logger.info(f"File '{remote_path}' newer on peer ({datetime.fromtimestamp(remote_meta['timestamp']).strftime('%H:%M:%S')}). Will download.")
                        elif local_meta['timestamp'] < remote_meta['timestamp']: # Should be covered by first condition, but good for clarity
                            files_to_download.append(remote_meta)
                            logger.info(f"File '{remote_path}' older locally ({datetime.fromtimestamp(local_meta['timestamp']).strftime('%H:%M:%S')}), peer has newer. Will download.")
                        elif local_meta['timestamp'] == remote_meta['timestamp']:
                            # Same timestamp, different hash - highly unlikely but can happen.
                            # For simplicity, we prioritize the remote peer's version in this case.
                            # A real system would need more robust conflict resolution (e.g., versioning).
                            files_to_download.append(remote_meta)
                            logger.warning(f"Conflict detected for '{remote_path}'. Timestamps identical but hashes differ. Prioritizing peer's version.")
                    # else: Hashes match, nothing to do (files are identical).

                # 3. Download necessary files
                for file_meta in files_to_download:
                    if not send_json_message(sock, 'FILE_REQUEST', {'path': file_meta['path']}):
                        logger.error(f"Failed to send FILE_REQUEST for '{file_meta['path']}' to {peer_id}.")
                        continue
                    
                    response = receive_json_message(sock)
                    if response and response.get('type') == 'FILE_DATA':
                        if not self._receive_file(sock, response['payload']):
                            logger.error(f"Failed to receive file '{file_meta['path']}' from {peer_id}.")
                        else:
                            # Update local metadata immediately after successful download
                            self.local_file_metadata[file_meta['path']] = self._get_local_file_metadata(file_meta['path'])
                    elif response and response.get('type') == 'FILE_NOT_FOUND':
                        logger.warning(f"Peer {peer_id} reported FILE_NOT_FOUND for '{file_meta['path']}'.")
                    else:
                        logger.error(f"Unexpected response during file request for '{file_meta['path']}' from {peer_id}: {response}")
                
                logger.info(f"Synchronization with {peer_id} complete. Files downloaded: {len(files_to_download)}")

            except socket.error as e:
                logger.error(f"Socket error during sync with {peer_id}: {e}")
                # Close and remove connection on error
                if peer_id in self.active_peer_connections:
                    self.active_peer_connections[peer_id].close()
                    del self.active_peer_connections[peer_id]
            except Exception as e:
                logger.error(f"Error during sync with {peer_id}: {e}", exc_info=True)
            finally:
                # For this simplified model, we close the socket after each sync round.
                # For continuous communication, the connection could be kept open.
                # However, re-establishing per sync simplifies state management.
                if peer_id in self.active_peer_connections:
                    self.active_peer_connections[peer_id].close()
                    del self.active_peer_connections[peer_id]
                logger.info(f"Disconnected from {peer_id} after sync round.")


    def _synchronization_loop(self):
        """
        The main synchronization loop that periodically initiates sync rounds with all known peers.
        This runs in a separate thread.
        """
        while self.running:
            time.sleep(SYNC_INTERVAL_SECONDS)
            if not self.running:
                break
            
            self._scan_local_files() # Ensure local state is fresh before initiating sync

            # Iterate over a copy of known peers as the set might be modified during sync
            peers_to_sync = list(self.peers) 
            for peer_host, peer_port in peers_to_sync:
                if (peer_host, peer_port) != (self.host, self.port): # Don't sync with self
                    self._synchronize_with_peer((peer_host, peer_port))
            
            logger.info("Sync loop completed a round with known peers.")

    def _on_file_system_event(self, event):
        """
        Callback function for watchdog file system events.
        Triggers an immediate scan of local files, but does NOT initiate an immediate sync
        to avoid overwhelming the network with frequent small changes. The periodic sync loop
        will pick up the changes.

        Args:
            event (FileSystemEvent): The watchdog event object.
        """
        if not self.running:
            return

        # We only care about file events within our shared directory
        # Watchdog events can sometimes report on temporary files or directories being created/deleted.
        # Filtering is important.
        if os.path.isdir(event.src_path):
            return

        # Ignore events that might be caused by our own sync operations (e.g., file writing)
        # This is a heuristic and can be tricky. For now, we trust the sync logic.

        relative_path = os.path.relpath(event.src_path, self.shared_dir)
        logger.debug(f"File system event detected: {event.event_type} - {relative_path}")

        # Re-scan local files immediately to reflect the change in our internal state
        self._scan_local_files() 
        
        logger.info(f"Local file change detected for '{relative_path}'. Updated internal state. "
                    "Next periodic sync round will propagate changes.")

    def _start_file_monitor(self):
        """
        Starts the watchdog file system monitor if `watchdog` is available.
        This allows for near real-time detection of local file changes.
        """
        if not WATCHDOG_AVAILABLE:
            return

        event_handler = FileChangeHandler(self._on_file_system_event)
        observer = Observer()
        observer.schedule(event_handler, self.shared_dir, recursive=True)
        observer.start()
        self.threads.append(observer) # Keep reference to observer for proper shutdown
        logger.info(f"Real-time file monitoring started for {self.shared_dir}")


    def start(self):
        """
        Starts all main components of the peer: the server for incoming connections,
        the synchronization loop, and optionally the file system monitor.
        """
        # Start server thread
        server_thread = threading.Thread(target=self._start_server, daemon=True)
        server_thread.name = "PeerServer"
        self.threads.append(server_thread)
        server_thread.start()

        # Start synchronization loop thread
        sync_thread = threading.Thread(target=self._synchronization_loop, daemon=True)
        sync_thread.name = "SyncLoop"
        self.threads.append(sync_thread)
        sync_thread.start()

        # Start file system monitor if watchdog is available
        if WATCHDOG_AVAILABLE:
            self._start_file_monitor()
        else:
            logger.warning("Watchdog is not available. File changes will only be detected during periodic scans.")

        logger.info(f"Peer {self.host}:{self.port} started.")

    def stop(self):
        """
        Stops all peer components gracefully and cleans up resources (sockets, threads).
        """
        logger.info(f"Stopping Peer {self.host}:{self.port}...")
        self.running = False # Signal all running loops to terminate

        # Close the server socket
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
                self.server_socket.close()
                logger.info("Server socket closed.")
            except OSError as e:
                logger.warning(f"Error shutting down server socket: {e}")

        # Close all active outgoing connections
        for peer_id, sock in list(self.active_peer_connections.items()):
            try:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
                logger.info(f"Closed active connection to {peer_id}")
            except OSError as e:
                logger.warning(f"Error closing connection to {peer_id}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error closing connection to {peer_id}: {e}")
            del self.active_peer_connections[peer_id] # Remove from tracking

        # Stop watchdog observer if it's running
        if WATCHDOG_AVAILABLE:
            for thread in self.threads:
                if isinstance(thread, Observer):
                    if thread.is_alive():
                        logger.info("Stopping Watchdog observer...")
                        thread.stop()
                        thread.join(timeout=5) # Wait for observer thread to finish
                        if thread.is_alive():
                            logger.warning("Watchdog observer thread did not terminate gracefully.")
                    break
        
        # Give a small moment for daemon threads to register the 'running = False'
        time.sleep(0.5)
        logger.info(f"Peer {self.host}:{self.port} stopped successfully.")

# --- Watchdog Event Handler (if enabled) ---
if WATCHDOG_AVAILABLE:
    class FileChangeHandler(FileSystemEventHandler):
        """
        Custom event handler for watchdog to process file system events.
        Filters for relevant file modification events.
        """
        def __init__(self, callback):
            super().__init__()
            self.callback = callback

        def on_any_event(self, event):
            """
            Called on any file system event. Filters and passes relevant events to the callback.
            """
            # Ignore directory events for simplicity, focus on actual files
            if event.is_directory:
                return 

            # Only trigger callback for relevant events that change file content or presence
            if event.event_type in ('created', 'modified', 'deleted', 'moved'):
                # Handle 'moved' events: watchdog reports two events, deleted (src) and created (dest)
                # For simplicity, we process both as individual file events.
                self.callback(event)

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decentralized P2P File Sharing and Synchronization System.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="The host address for this peer to listen on (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"The port for this peer to listen on (default: {DEFAULT_PORT}).")
    parser.add_argument("--dir", type=str, default=DEFAULT_SHARED_DIR,
                        help=f"The base directory to share and synchronize. If default, it will be '{DEFAULT_SHARED_DIR}_<port>' (default: '{DEFAULT_SHARED_DIR}').")
    parser.add_argument("--peers", type=str, nargs='*',
                        help="List of initial known peers in 'host:port' format (e.g., '127.0.0.1:9001 127.0.0.1:9002').")
    
    args = parser.parse_args()

    # Parse known peers from command line arguments
    known_peers_list = []
    if args.peers:
        for peer_str in args.peers:
            try:
                p_host, p_port = peer_str.split(':')
                known_peers_list.append((p_host, int(p_port)))
            except ValueError:
                logger.error(f"Invalid peer format: '{peer_str}'. Expected 'host:port'. Skipping this peer.")
    
    # Ensure shared directory is unique for each peer instance if default is used
    shared_dir = args.dir
    if args.dir == DEFAULT_SHARED_DIR:
        shared_dir = f"{DEFAULT_SHARED_DIR}_{args.port}"

    my_peer = None
    try:
        my_peer = Peer(args.host, args.port, shared_dir, known_peers=known_peers_list)
        my_peer.start()

        logger.info("\n---------------------------------------------------------")
        logger.info(f"P2P Peer running on {my_peer.host}:{my_peer.port}")
        logger.info(f"Sharing directory: {my_peer.shared_dir}")
        logger.info("Type 'status' for peer info, 'exit' to stop, or press Ctrl+C.")
        logger.info("---------------------------------------------------------")

        # Keep the main thread alive to listen for user input
        while my_peer.running:
            try:
                command = input("> ").strip().lower()
                if command == 'exit':
                    my_peer.running = False # Signal all threads to stop
                elif command == 'status':
                    my_peer._scan_local_files() # Refresh local view for status
                    logger.info("--- Peer Status ---")
                    logger.info(f"My address: {my_peer.host}:{my_peer.port}")
                    logger.info(f"Shared Directory: {my_peer.shared_dir}")
                    logger.info(f"Total Local Files: {len(my_peer.local_file_metadata)}")
                    for p, m in my_peer.local_file_metadata.items():
                        logger.info(f"  - {p} (Hash: {m['hash'][:8]}..., Size: {m['size']} bytes, Modified: {datetime.fromtimestamp(m['timestamp']).strftime('%Y-%m-%d %H:%M:%S')})")
                    logger.info(f"Known Peers: {my_peer.peers}")
                    logger.info(f"Active Outgoing Connections: {list(my_peer.active_peer_connections.keys())}")
                    logger.info("-------------------")
                else:
                    logger.info("Unknown command. Use 'status' or 'exit'.")
            except EOFError: # Handles Ctrl+D (End of File)
                logger.info("EOF detected. Exiting.")
                my_peer.running = False
            except KeyboardInterrupt: # Handles Ctrl+C (Interrupt)
                logger.info("KeyboardInterrupt detected. Exiting.")
                my_peer.running = False
            
            # Small sleep to prevent busy-waiting in case of tight input loop
            time.sleep(0.1)

    except Exception as e:
        logger.critical(f"An unhandled critical error occurred in the main process: {e}", exc_info=True)
    finally:
        # Ensure cleanup happens even if an error occurs
        if my_peer:
            my_peer.stop()
        logger.info("P2P Peer application terminated.")
```