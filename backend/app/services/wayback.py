"""Wayback Machine API integration service"""
import re
import httpx
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from collections import defaultdict


class WaybackService:
    """Service for interacting with the Wayback Machine CDX API"""

    CDX_API_URL = "http://web.archive.org/cdx/search/cdx"
    WAYBACK_BASE_URL = "http://web.archive.org/web"

    # A CDX query over a whole domain is a scan, not a lookup, and the server
    # streams nothing until it has finished: a large site takes minutes. Sixty
    # seconds was enough for a small blog and silently failed on anything
    # bigger -- httpx raises ReadTimeout, whose str() is empty, so the API
    # reported "Failed to query Wayback Machine: " with no reason. Connecting
    # stays quick; only the read is allowed to be slow.
    CDX_TIMEOUT = httpx.Timeout(600.0, connect=30.0)

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=self.CDX_TIMEOUT)

    async def close(self):
        await self.client.aclose()

    async def get_snapshot_count(self, domain: str) -> int:
        """
        Get the true total count of snapshots for a domain.
        Uses the CDX API's showNumPages feature to calculate total without fetching all data.
        """
        params = {
            "url": f"{domain}/*",
            "output": "json",
            "fl": "timestamp",  # Minimal field to reduce response size
            "collapse": "digest",
            "showNumPages": "true",
        }

        try:
            response = await self.client.get(self.CDX_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # When showNumPages=true, the API returns the number of pages
            # Default page size is 10000, so total = pages * 10000
            # However, this is an approximation - last page may have fewer results
            if isinstance(data, int):
                num_pages = data
            elif isinstance(data, list) and len(data) > 0:
                num_pages = int(data[0]) if isinstance(data[0], (int, str)) else 1
            else:
                return 0

            # For a more accurate count, we need to fetch the actual data
            # But since showNumPages gives us pages, we estimate the max
            # We'll use a follow-up request without limit to get exact count
            return await self._get_exact_count(domain)

        except httpx.HTTPError:
            return 0

    async def _get_exact_count(self, domain: str) -> int:
        """
        Get exact snapshot count by requesting just the count without full data.
        Uses a request with minimal fields to get accurate count efficiently.
        """
        params = {
            "url": f"{domain}/*",
            "output": "json",
            "fl": "timestamp",  # Minimal field
            "collapse": "digest",
            # No limit - get all results to count them
        }

        try:
            response = await self.client.get(self.CDX_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if not data or len(data) < 2:
                return 0

            # First row is headers, rest are data
            return len(data) - 1

        except httpx.HTTPError:
            return 0

    async def search_domain(self, domain: str, limit: int = 10000) -> Dict[str, Any]:
        """
        Search for all snapshots of a domain in the Wayback Machine.
        Returns summary info and sample URLs.
        """
        params = {
            "url": f"{domain}/*",
            "output": "json",
            "fl": "timestamp,original,mimetype,statuscode,digest,length",
            "collapse": "digest",  # Deduplicate by content hash
            "limit": limit,
        }

        try:
            response = await self.client.get(self.CDX_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if not data or len(data) < 2:
                return {
                    "domain": domain,
                    "total_snapshots": 0,
                    "date_range_start": None,
                    "date_range_end": None,
                    "sample_urls": [],
                    "mimetypes": [],
                }

            # First row is headers
            headers = data[0]
            rows = data[1:]

            # Parse data
            timestamps = []
            urls = set()
            mimetypes = set()

            for row in rows:
                record = dict(zip(headers, row))
                timestamps.append(record["timestamp"])
                urls.add(record["original"])
                mimetypes.add(record["mimetype"])

            timestamps.sort()

            # Get accurate total count (not limited by the sample limit)
            total_count = await self.get_snapshot_count(domain)
            # Fall back to len(rows) if count request fails
            if total_count == 0 and len(rows) > 0:
                total_count = len(rows)

            return {
                "domain": domain,
                "total_snapshots": total_count,
                "date_range_start": timestamps[0] if timestamps else None,
                "date_range_end": timestamps[-1] if timestamps else None,
                "sample_urls": list(urls)[:50],  # Return first 50 unique URLs
                "mimetypes": sorted(list(mimetypes)),
            }

        except httpx.HTTPError as e:
            raise Exception(f"Failed to query Wayback Machine: {str(e)}")

    async def get_all_snapshots(
        self,
        domain: str,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        mimetypes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all snapshots for a domain with optional filters.
        Returns list of snapshot records.
        """
        params = {
            "url": f"{domain}/*",
            "output": "json",
            "fl": "timestamp,original,mimetype,statuscode,digest,length",
            "collapse": "digest",
        }

        if date_start:
            params["from"] = date_start
        if date_end:
            params["to"] = date_end
        if mimetypes:
            # One regex filter. The CDX API ANDs repeated filter params, so one
            # filter per type matched nothing as soon as two types were picked.
            alts = "|".join(re.escape(mt) for mt in mimetypes)
            params["filter"] = f"mimetype:^({alts})$"

        try:
            response = await self.client.get(self.CDX_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if not data or len(data) < 2:
                return []

            headers = data[0]
            rows = data[1:]

            snapshots = []
            for row in rows:
                record = dict(zip(headers, row))
                snapshots.append({
                    "timestamp": record["timestamp"],
                    "original_url": record["original"],
                    "mimetype": record["mimetype"],
                    "status_code": record["statuscode"],
                    "digest": record["digest"],
                    "length": int(record["length"]) if record["length"] else 0,
                })

            return snapshots

        except httpx.HTTPError as e:
            raise Exception(f"Failed to fetch snapshots: {str(e)}")

    def filter_by_frequency(
        self, snapshots: List[Dict[str, Any]], frequency: str
    ) -> List[Dict[str, Any]]:
        """
        Filter snapshots by frequency (keep one per time period per URL).
        frequency: all, yearly, monthly, weekly, daily
        """
        if frequency == "all":
            return snapshots

        # Group by URL and time period
        grouped = defaultdict(dict)

        for snapshot in snapshots:
            url = snapshot["original_url"]
            ts = snapshot["timestamp"]

            if frequency == "yearly":
                period = ts[:4]  # YYYY
            elif frequency == "monthly":
                period = ts[:6]  # YYYYMM
            elif frequency == "weekly":
                # Calculate week number
                try:
                    dt = datetime.strptime(ts[:8], "%Y%m%d")
                    period = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
                except ValueError:
                    period = ts[:6]
            elif frequency == "daily":
                period = ts[:8]  # YYYYMMDD
            else:
                period = ts

            key = (url, period)
            if key not in grouped:
                grouped[key] = snapshot

        return list(grouped.values())

    def build_wayback_url(self, timestamp: str, original_url: str) -> str:
        """Build the Wayback Machine URL for a specific snapshot"""
        return f"{self.WAYBACK_BASE_URL}/{timestamp}id_/{original_url}"

    async def download_snapshot(self, wayback_url: str) -> bytes:
        """Download content from a Wayback Machine URL"""
        try:
            response = await self.client.get(wayback_url, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as e:
            raise Exception(f"Failed to download {wayback_url}: {str(e)}")


# Global instance
wayback_service = WaybackService()
