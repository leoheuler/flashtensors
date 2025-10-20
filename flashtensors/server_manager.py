import os
import subprocess
import threading
import time
from typing import Optional
from .utils import init_logger
from .config import get_config, get_server_config, is_server_running

logger = init_logger(__name__)

class ServerManager:
    _instance: Optional['ServerManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.supervisord_process: Optional[subprocess.Popen] = None
        self._server_started = False
        self._lock = threading.Lock()
        self.config_file = os.path.join(os.path.dirname(__file__), 'supervisord.conf')

    def start_server_if_needed(self) -> bool:
        with self._lock:
            config = get_config()
            host = config["server_host"]
            port = config["server_port"]

            if is_server_running():
                logger.info(f"gRPC server already running on {host}:{port}")
                return True

            if self._server_started:
                logger.info("Server startup already initiated")
                return True

            logger.info(f"Starting TeilEngine gRPC server via supervisord on {host}:{port}...")
            
            if not self._start_supervisord():
                return False

            if not self._start_storage_server():
                return False

            start_time = time.time()
            timeout = 30
            while time.time() - start_time < timeout:
                if is_server_running():
                    logger.info(f"✅ gRPC server started successfully on {host}:{port}")
                    self._server_started = True
                    return True
                time.sleep(0.5)
            
            logger.error(f"❌ Failed to start gRPC server after {timeout} seconds")
            return False

    def _start_supervisord(self) -> bool:
        logger.info("Environment variables before starting supervisord:")
        for key in ["TEILENGINE_HOST", "TEILENGINE_PORT", "TEILENGINE_STORAGE_PATH", 
                    "TEILENGINE_NUM_THREADS", "TEILENGINE_CHUNK_SIZE", "TEILENGINE_MEM_POOL_SIZE"]:
            logger.info(f"  {key}={os.environ.get(key, 'NOT SET')}")

        try:
            if self._is_supervisord_running():
                logger.info("Supervisord already running")
                return True
            
            logger.info("Starting supervisord...")
            cmd = ["supervisord", "-c", self.config_file]
            current_env = os.environ.copy()
            
            self.supervisord_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                text=True,
                env=current_env
            )

            time.sleep(2)

            stdout, stderr = self.supervisord_process.communicate(timeout=10)
            if self.supervisord_process.returncode == 0:
                logger.info("Supervisord command completed successfully (daemonized)")
                return True

            logger.error(f"❌ Failed to start supervisord: {stderr} stdout: {stdout}")
            return False
        except Exception as e:
            logger.error(f"Failed to start supervisord: {e}")
            return False
    
    def _start_storage_server(self) -> bool:
        try:
            logger.info("Starting TeilEngine storage server via supervisorctl...")
            cmd = ["supervisorctl", "-c", self.config_file, "start", "flashtensors_storage_server"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ Storage server started via supervisorctl")
                return True
            else:
                logger.error(f"❌ Failed to start storage server: {result.stderr}")
                logger.error(f"stdout: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Storage server start timed out after 30 seconds")
            return False
        except Exception as e:
            logger.error(f"Failed to start storage server via supervisorctl: {e}")
            return False
    
    def _is_supervisord_running(self) -> bool:
        try:
            result = subprocess.run(
                ["supervisorctl", "-c", self.config_file, "status"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
    
    def stop_server(self):
        with self._lock:
            try:
                logger.info("Stopping TeilEngine storage server...")
                
                # Stop the storage server via supervisorctl
                cmd = ["supervisorctl", "-c", self.config_file, "stop", "flashtensors_storage_server"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Storage server stopped via supervisorctl")
                else:
                    logger.warning(f"Failed to stop storage server gracefully: {result.stderr}")
                
                # Stop supervisord
                logger.info("Stopping supervisord...")
                cmd = ["supervisorctl", "-c", self.config_file, "shutdown"]
                subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if self.supervisord_process:
                    try:
                        self.supervisord_process.terminate()
                        self.supervisord_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.supervisord_process.kill()
                    finally:
                        self.supervisord_process = None
                
                logger.info("✅ TeilEngine server stack stopped")
                
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
            finally:
                self._server_started = False

    def ensure_server_running(self) -> bool:
        return self.start_server_if_needed()

_server_manager: Optional[ServerManager] = None

def get_server_manager() -> ServerManager:
    global _server_manager
    if _server_manager is None:
        _server_manager = ServerManager()
    return _server_manager

def ensure_server_running() -> bool:
    return get_server_manager().ensure_server_running()
