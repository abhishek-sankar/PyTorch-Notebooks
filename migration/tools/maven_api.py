"""
Maven Central API Client for Java Migration System

Provides integration with Maven Central API for dependency analysis,
version checking, and compatibility validation.
"""

import requests
import time
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from urllib.parse import quote

logger = logging.getLogger(__name__)


class MavenCentralAPI:
    """
    Client for interacting with Maven Central API.
    
    Provides functionality to:
    - Search for artifacts and versions
    - Get latest versions of dependencies
    - Check Java compatibility
    - Analyze dependency vulnerabilities
    """
    
    def __init__(self, base_url: str = "https://search.maven.org/solrsearch/select"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Java-Migration-System/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Caching
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=1)
    
    def get_latest_version(self, group_id: str, artifact_id: str) -> str:
        """
        Get the latest version of a Maven artifact.
        
        Args:
            group_id: Maven group ID
            artifact_id: Maven artifact ID
            
        Returns:
            Latest version string or "unknown" if not found
        """
        cache_key = f"{group_id}:{artifact_id}:latest"
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result["version"]
        
        logger.info(f"Fetching latest version for {group_id}:{artifact_id}")
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Build search query
            query = f'g:"{group_id}" AND a:"{artifact_id}"'
            params = {
                'q': query,
                'core': 'gav',
                'rows': 1,
                'wt': 'json'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            docs = data.get('response', {}).get('docs', [])
            
            if docs:
                latest_version = docs[0].get('v', 'unknown')
                
                # Cache the result
                self._cache_result(cache_key, {
                    "version": latest_version,
                    "timestamp": datetime.now()
                })
                
                logger.info(f"Latest version for {group_id}:{artifact_id} is {latest_version}")
                return latest_version
            else:
                logger.warning(f"No versions found for {group_id}:{artifact_id}")
                return "unknown"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch latest version for {group_id}:{artifact_id}: {e}")
            return "unknown"
        except Exception as e:
            logger.error(f"Unexpected error fetching version: {e}")
            return "unknown"
    
    def get_all_versions(self, group_id: str, artifact_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get all versions of a Maven artifact.
        
        Args:
            group_id: Maven group ID
            artifact_id: Maven artifact ID
            limit: Maximum number of versions to return
            
        Returns:
            List of version information dictionaries
        """
        cache_key = f"{group_id}:{artifact_id}:all_versions:{limit}"
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result["versions"]
        
        logger.info(f"Fetching all versions for {group_id}:{artifact_id}")
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Build search query
            query = f'g:"{group_id}" AND a:"{artifact_id}"'
            params = {
                'q': query,
                'core': 'gav',
                'rows': limit,
                'wt': 'json'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            docs = data.get('response', {}).get('docs', [])
            
            versions = []
            for doc in docs:
                version_info = {
                    "version": doc.get('v', 'unknown'),
                    "timestamp": doc.get('timestamp', 0),
                    "group_id": doc.get('g', group_id),
                    "artifact_id": doc.get('id', artifact_id)
                }
                versions.append(version_info)
            
            # Cache the result
            self._cache_result(cache_key, {
                "versions": versions,
                "timestamp": datetime.now()
            })
            
            logger.info(f"Found {len(versions)} versions for {group_id}:{artifact_id}")
            return versions
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch versions for {group_id}:{artifact_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching versions: {e}")
            return []
    
    def check_java_compatibility(self, group_id: str, artifact_id: str, version: str, java_version: str = "21") -> Dict[str, Any]:
        """
        Check if an artifact version is compatible with a Java version.
        
        Args:
            group_id: Maven group ID
            artifact_id: Maven artifact ID
            version: Artifact version
            java_version: Target Java version
            
        Returns:
            Dictionary with compatibility information
        """
        logger.info(f"Checking Java {java_version} compatibility for {group_id}:{artifact_id}:{version}")
        
        # This is a heuristic-based approach as Maven Central doesn't directly provide
        # Java compatibility information
        
        try:
            # Get artifact details
            artifact_info = self.get_artifact_info(group_id, artifact_id, version)
            
            compatibility = {
                "compatible": True,  # Default assumption
                "confidence": "medium",
                "issues": [],
                "recommendations": []
            }
            
            # Heuristic checks based on common patterns
            
            # 1. Check version patterns that indicate Java requirements
            version_lower = version.lower()
            
            # Known patterns for different Java versions
            if any(pattern in version_lower for pattern in ['jdk6', 'java6', 'j6']):
                if int(java_version) > 8:
                    compatibility["issues"].append("Artifact built for Java 6, may have compatibility issues")
                    compatibility["confidence"] = "low"
            
            if any(pattern in version_lower for pattern in ['jdk7', 'java7', 'j7']):
                if int(java_version) > 11:
                    compatibility["issues"].append("Artifact built for Java 7, may have compatibility issues")
                    compatibility["confidence"] = "low"
            
            # 2. Check for known problematic artifacts
            problematic_artifacts = {
                "javax.xml.bind:jaxb-api": "Use jakarta.xml.bind:jakarta.xml.bind-api for Java 11+",
                "javax.annotation:javax.annotation-api": "Use jakarta.annotation:jakarta.annotation-api for Java 11+",
                "com.sun.xml.bind:jaxb-core": "JAXB removed from JDK 11+, need explicit dependency"
            }
            
            artifact_key = f"{group_id}:{artifact_id}"
            if artifact_key in problematic_artifacts:
                compatibility["issues"].append(problematic_artifacts[artifact_key])
                compatibility["compatible"] = False
                compatibility["confidence"] = "high"
            
            # 3. Check artifact age (very old artifacts may have issues)
            if artifact_info.get("timestamp"):
                artifact_date = datetime.fromtimestamp(artifact_info["timestamp"] / 1000)
                age_years = (datetime.now() - artifact_date).days / 365
                
                if age_years > 5:
                    compatibility["issues"].append(f"Artifact is {age_years:.1f} years old, may need updating")
                    compatibility["confidence"] = "low"
            
            # 4. Provide general recommendations
            if int(java_version) >= 17:
                compatibility["recommendations"].append("Ensure all dependencies support modules if using JPMS")
                compatibility["recommendations"].append("Check for SecurityManager usage (deprecated in Java 17)")
            
            if int(java_version) >= 21:
                compatibility["recommendations"].append("Verify compatibility with virtual threads if used")
                compatibility["recommendations"].append("Check for usage of deprecated/removed APIs")
            
            return compatibility
            
        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return {
                "compatible": True,  # Assume compatible on error
                "confidence": "unknown",
                "error": str(e),
                "issues": [],
                "recommendations": []
            }
    
    def get_artifact_info(self, group_id: str, artifact_id: str, version: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific artifact version.
        
        Args:
            group_id: Maven group ID
            artifact_id: Maven artifact ID
            version: Artifact version
            
        Returns:
            Dictionary with artifact information
        """
        cache_key = f"{group_id}:{artifact_id}:{version}:info"
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result["info"]
        
        logger.info(f"Fetching info for {group_id}:{artifact_id}:{version}")
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Build search query
            query = f'g:"{group_id}" AND a:"{artifact_id}" AND v:"{version}"'
            params = {
                'q': query,
                'core': 'gav',
                'rows': 1,
                'wt': 'json'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            docs = data.get('response', {}).get('docs', [])
            
            if docs:
                info = docs[0]
                
                # Cache the result
                self._cache_result(cache_key, {
                    "info": info,
                    "timestamp": datetime.now()
                })
                
                return info
            else:
                logger.warning(f"No info found for {group_id}:{artifact_id}:{version}")
                return {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch artifact info: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching artifact info: {e}")
            return {}
    
    def search_artifacts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for artifacts using a text query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of artifact information dictionaries
        """
        logger.info(f"Searching artifacts with query: {query}")
        
        try:
            # Rate limiting
            self._rate_limit()
            
            params = {
                'q': query,
                'rows': limit,
                'wt': 'json'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            docs = data.get('response', {}).get('docs', [])
            
            results = []
            for doc in docs:
                result = {
                    "group_id": doc.get('g', ''),
                    "artifact_id": doc.get('a', ''),
                    "version": doc.get('v', ''),
                    "timestamp": doc.get('timestamp', 0),
                    "version_count": doc.get('versionCount', 0)
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} artifacts for query: {query}")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return []
    
    def get_dependency_updates(self, dependencies: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Check for updates for a list of dependencies.
        
        Args:
            dependencies: List of dependency dictionaries with group_id, artifact_id, version
            
        Returns:
            List of update information dictionaries
        """
        logger.info(f"Checking updates for {len(dependencies)} dependencies")
        
        updates = []
        
        for dep in dependencies:
            group_id = dep.get("group_id")
            artifact_id = dep.get("artifact_id")
            current_version = dep.get("version")
            
            if not all([group_id, artifact_id, current_version]):
                continue
            
            try:
                latest_version = self.get_latest_version(group_id, artifact_id)
                
                update_info = {
                    "group_id": group_id,
                    "artifact_id": artifact_id,
                    "current_version": current_version,
                    "latest_version": latest_version,
                    "update_available": latest_version != current_version and latest_version != "unknown",
                    "compatibility": None
                }
                
                # Check Java 21 compatibility if update is available
                if update_info["update_available"]:
                    compatibility = self.check_java_compatibility(group_id, artifact_id, latest_version, "21")
                    update_info["compatibility"] = compatibility
                
                updates.append(update_info)
                
            except Exception as e:
                logger.warning(f"Failed to check updates for {group_id}:{artifact_id}: {e}")
                continue
        
        # Filter to only updates that are available and compatible
        compatible_updates = [
            update for update in updates 
            if update["update_available"] and 
            (not update["compatibility"] or update["compatibility"].get("compatible", True))
        ]
        
        logger.info(f"Found {len(compatible_updates)} compatible updates out of {len(updates)} dependencies")
        return updates
    
    def clear_cache(self):
        """Clear the internal cache"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_keys": list(self.cache.keys())
        }
    
    # Private helper methods
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from cache if not expired"""
        if key not in self.cache:
            return None
        
        cached_data = self.cache[key]
        cache_time = cached_data.get("cache_timestamp")
        
        if cache_time and datetime.now() - cache_time < self.cache_ttl:
            return cached_data
        else:
            # Remove expired entry
            del self.cache[key]
            return None
    
    def _cache_result(self, key: str, data: Dict[str, Any]):
        """Cache a result with timestamp"""
        data["cache_timestamp"] = datetime.now()
        self.cache[key] = data