#!/usr/bin/env python
"""
Utils4BurnScript - Utilities for BURN Script orchestration
===========================================================
Handles:
- Subprocess-based mapping with live logging
- HTML dashboard generation with auto-refresh
- Job management and progress tracking
"""
import subprocess
import json
import sys
from pathlib import Path
from time import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import threading


# =============================================================================
# MAPPING WORKER SCRIPT (runs in subprocess)
# =============================================================================
MAPPING_WORKER_SCRIPT = '''
import sys
import json
import pycolmap
from pathlib import Path
from time import time
from datetime import datetime

def run_mapping(db_path, img_path, out_path, opts_json, log_file, job_id):
    """Run pycolmap mapping and write progress to log file."""
    log = {
        "job_id": job_id,
        "status": "running",
        "success": False,
        "num_registered": 0,
        "num_points3d": 0,
        "elapsed": 0,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": []
    }
    start = time()
    
    def update_log(msg):
        log["elapsed"] = time() - start
        log["updated_at"] = datetime.now().isoformat()
        log["messages"].append({"t": f"{log['elapsed']:.1f}s", "msg": msg})
        if len(log["messages"]) > 50:
            log["messages"] = log["messages"][-50:]
        with open(log_file, "w") as f:
            json.dump(log, f, indent=2)
    
    try:
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        
        update_log("Initializing mapper options")
        opts = pycolmap.IncrementalPipelineOptions()
        
        if opts_json and opts_json != "{}":
            mapper_options = json.loads(opts_json)
            if "min_num_matches" in mapper_options:
                opts.min_num_matches = mapper_options["min_num_matches"]
            if "multiple_models" in mapper_options:
                opts.multiple_models = mapper_options["multiple_models"]
            if "extract_colors" in mapper_options:
                opts.extract_colors = mapper_options["extract_colors"]
        
        update_log("Starting incremental mapping...")
        pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(img_path),
            output_path=str(out_path),
            options=opts
        )
        
        update_log("Reading reconstruction stats")
        recon_path = out_path / "0"
        if recon_path.exists():
            try:
                recon = pycolmap.Reconstruction(str(recon_path))
                log["num_registered"] = recon.num_reg_images()
                log["num_points3d"] = recon.num_points3D()
                log["success"] = True
                update_log(f"Done: {log['num_registered']} images, {log['num_points3d']} points")
            except Exception as e:
                log["success"] = True
                update_log(f"Reconstruction exists but stats unavailable: {e}")
        else:
            update_log("No reconstruction created (sparse/0 not found)")
        
        log["status"] = "completed"
        
    except Exception as e:
        log["error"] = str(e)
        log["status"] = "failed"
        update_log(f"ERROR: {str(e)}")
    
    log["elapsed"] = time() - start
    log["updated_at"] = datetime.now().isoformat()
    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    db_path, img_path, out_path, opts_json, log_file, job_id = sys.argv[1:7]
    run_mapping(db_path, img_path, out_path, opts_json, log_file, job_id)
'''


# =============================================================================
# MAPPING JOB DATACLASS
# =============================================================================
@dataclass
class MappingJob:
    """Tracks a background mapping job."""
    job_id: str
    name: str  # Human-readable name (e.g., "group_001/cam0")
    database_path: Path
    image_path: Path
    output_path: Path
    log_file: Path
    process: subprocess.Popen = None
    started_at: float = 0
    finished: bool = False
    result: dict = field(default_factory=dict)


# =============================================================================
# MAPPING JOB MANAGER
# =============================================================================
class MappingJobManager:
    """Manages background mapping jobs with live dashboard."""
    
    def __init__(self, dashboard_dir: Path, update_interval: float = 5.0):
        self.jobs: Dict[str, MappingJob] = {}
        self.dashboard_dir = Path(dashboard_dir)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self._job_counter = 0
        self._stop_updater = threading.Event()
        self._update_interval = update_interval
        self._updater_thread = None
        self._create_dashboard_html()
        self._update_dashboard_status()  # Create initial status file
    
    def _start_background_updater(self):
        """Start background thread that periodically updates dashboard."""
        if self._updater_thread is not None and self._updater_thread.is_alive():
            return  # Already running
        
        self._stop_updater.clear()
        
        def updater_loop():
            while not self._stop_updater.is_set():
                try:
                    # Check all jobs and update their status
                    for job_id, job in list(self.jobs.items()):
                        if not job.finished:
                            poll = job.process.poll()
                            if poll is not None:
                                # Job finished - read final log
                                job.finished = True
                                job.result = self.read_job_log(job_id)
                                job.result['elapsed'] = time() - job.started_at
                                self._cleanup_job_files(job)
                    
                    self._update_dashboard_status()
                except Exception as e:
                    pass  # Don't crash the updater thread
                
                self._stop_updater.wait(self._update_interval)
        
        self._updater_thread = threading.Thread(target=updater_loop, daemon=True)
        self._updater_thread.start()
    
    def _stop_background_updater(self):
        """Stop the background updater thread."""
        self._stop_updater.set()
        if self._updater_thread is not None:
            self._updater_thread.join(timeout=2.0)
    
    def start_job(self, name: str, database_path: Path, image_path: Path, 
                  output_path: Path, mapper_options: dict = None) -> str:
        """Start a mapping job in background subprocess."""
        self._job_counter += 1
        job_id = f"map_{self._job_counter:03d}"
        
        # Create log file in output directory
        output_path.mkdir(parents=True, exist_ok=True)
        log_file = output_path / '_mapping_log.json'
        
        # Write initial log
        initial_log = {
            "job_id": job_id,
            "name": name,
            "status": "starting",
            "success": False,
            "num_registered": 0,
            "num_points3d": 0,
            "elapsed": 0,
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [{"t": "0.0s", "msg": "Job queued"}]
        }
        with open(log_file, 'w') as f:
            json.dump(initial_log, f, indent=2)
        
        # Write worker script
        worker_script = output_path / '_mapping_worker.py'
        with open(worker_script, 'w') as f:
            f.write(MAPPING_WORKER_SCRIPT)
        
        # Prepare options
        opts_json = json.dumps(mapper_options) if mapper_options else "{}"
        
        # Create job
        job = MappingJob(
            job_id=job_id,
            name=name,
            database_path=Path(database_path),
            image_path=Path(image_path),
            output_path=Path(output_path),
            log_file=log_file,
            started_at=time()
        )
        
        # Start subprocess
        job.process = subprocess.Popen(
            [sys.executable, str(worker_script),
             str(database_path), str(image_path), str(output_path),
             opts_json, str(log_file), job_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.jobs[job_id] = job
        print(f"    [BG] Started {job_id}: {name} (PID: {job.process.pid})")
        
        # Start background updater if not already running
        self._start_background_updater()
        
        # Update dashboard
        self._update_dashboard_status()
        
        return job_id
    
    def read_job_log(self, job_id: str) -> dict:
        """Read the current log for a job."""
        if job_id not in self.jobs:
            return {"error": f"Unknown job: {job_id}"}
        
        job = self.jobs[job_id]
        if job.log_file.exists():
            try:
                with open(job.log_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"status": "unknown", "job_id": job_id}
    
    def check_job(self, job_id: str, timeout: float = 0) -> Optional[dict]:
        """Check if job is done. Returns result if finished, None if running."""
        if job_id not in self.jobs:
            return {"error": f"Unknown job: {job_id}"}
        
        job = self.jobs[job_id]
        
        if job.finished:
            return job.result
        
        # Check subprocess
        poll = job.process.poll()
        elapsed = time() - job.started_at
        
        # Check timeout
        if timeout > 0 and elapsed > timeout and poll is None:
            print(f"    [TIMEOUT] Killing {job_id} after {elapsed:.0f}s")
            job.process.kill()
            job.process.wait()
            job.finished = True
            job.result = {
                "job_id": job_id,
                "name": job.name,
                "status": "timeout",
                "success": False,
                "error": f"Timeout after {elapsed:.0f}s",
                "elapsed": elapsed,
                "num_registered": 0,
                "num_points3d": 0
            }
            self._cleanup_job_files(job)
            self._update_dashboard_status()
            return job.result
        
        if poll is not None:
            # Process finished
            job.finished = True
            job.result = self.read_job_log(job_id)
            job.result['elapsed'] = elapsed
            self._cleanup_job_files(job)
            self._update_dashboard_status()
            return job.result
        
        return None  # Still running
    
    def _cleanup_job_files(self, job: MappingJob):
        """Remove temporary worker script (keep log for debugging)."""
        worker_script = job.output_path / '_mapping_worker.py'
        if worker_script.exists():
            worker_script.unlink()
    
    def get_all_status(self) -> List[dict]:
        """Get status of all jobs for dashboard."""
        statuses = []
        for job_id, job in self.jobs.items():
            if job.finished:
                status = job.result.copy()
            else:
                status = self.read_job_log(job_id)
                status['elapsed'] = time() - job.started_at
            status['job_id'] = job_id
            status['name'] = job.name
            statuses.append(status)
        return statuses
    
    def _update_dashboard_status(self):
        """Write current status to dashboard JSON file."""
        status_file = self.dashboard_dir / '_dashboard_status.json'
        data = {
            "updated_at": datetime.now().isoformat(),
            "total_jobs": len(self.jobs),
            "completed": sum(1 for j in self.jobs.values() if j.finished),
            "running": sum(1 for j in self.jobs.values() if not j.finished),
            "jobs": self.get_all_status()
        }
        with open(status_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _create_dashboard_html(self):
        """Create the auto-refreshing HTML dashboard."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Mapping Jobs Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d4ff; margin-bottom: 5px; }
        .summary { background: #16213e; padding: 15px; border-radius: 8px; margin: 15px 0; }
        .summary span { margin-right: 20px; }
        .running { color: #ffd700; }
        .completed { color: #00ff88; }
        .failed { color: #ff4757; }
        .timeout { color: #ff8c00; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th { background: #0f3460; color: #00d4ff; padding: 12px 8px; text-align: left; }
        td { padding: 10px 8px; border-bottom: 1px solid #333; }
        tr:hover { background: #16213e; }
        .status-running { background: #ffd70033; }
        .status-completed { background: #00ff8833; }
        .status-failed { background: #ff475733; }
        .status-timeout { background: #ff8c0033; }
        .log-box { max-height: 100px; overflow-y: auto; font-size: 11px; 
                   background: #0a0a15; padding: 5px; border-radius: 4px; }
        .refresh-info { color: #666; font-size: 12px; }
        #lastUpdate { color: #00d4ff; }
    </style>
</head>
<body>
    <h1>Mapping Jobs Dashboard</h1>
    <p class="refresh-info">Auto-refresh: 3s | Last update: <span id="lastUpdate">-</span> | 
       <small style="color:#888">If not loading, run: <code>python -m http.server 8000</code> in this folder</small></p>
    
    <div class="summary" id="summary">Loading...</div>
    
    <table>
        <thead>
            <tr>
                <th>Job ID</th>
                <th>Name</th>
                <th>Status</th>
                <th>Elapsed</th>
                <th>Registered</th>
                <th>3D Points</th>
                <th>Recent Log</th>
            </tr>
        </thead>
        <tbody id="jobsTable">
            <tr><td colspan="7">Loading...</td></tr>
        </tbody>
    </table>

    <script>
        function formatTime(seconds) {
            if (!seconds) return '-';
            if (seconds < 60) return seconds.toFixed(1) + 's';
            if (seconds < 3600) return (seconds / 60).toFixed(1) + 'm';
            return (seconds / 3600).toFixed(2) + 'h';
        }
        
        function getStatusClass(status) {
            if (status === 'completed') return 'completed';
            if (status === 'failed') return 'failed';
            if (status === 'timeout') return 'timeout';
            return 'running';
        }
        
        function loadJSON(url, callback) {
            // Use XMLHttpRequest for better file:// compatibility
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function() {
                if (xhr.readyState === 4) {
                    if (xhr.status === 200 || xhr.status === 0) {
                        try {
                            var data = JSON.parse(xhr.responseText);
                            callback(null, data);
                        } catch (e) {
                            callback('JSON parse error: ' + e.message, null);
                        }
                    } else {
                        callback('HTTP error: ' + xhr.status, null);
                    }
                }
            };
            xhr.onerror = function() {
                callback('Network error', null);
            };
            try {
                xhr.open('GET', url + '?' + Date.now(), true);
                xhr.send();
            } catch (e) {
                callback('Request error: ' + e.message, null);
            }
        }
        
        function updateDashboard() {
            loadJSON('_dashboard_status.json', function(err, data) {
                if (err) {
                    document.getElementById('summary').innerHTML = 
                        '<span class="failed">Error loading status: ' + err + '</span>' +
                        '<br><small>If using file://, try: python -m http.server 8000</small>';
                    return;
                }
                
                document.getElementById('lastUpdate').textContent = 
                    new Date(data.updated_at).toLocaleTimeString();
                
                let successCount = data.jobs.filter(j => j.success).length;
                let failedCount = data.jobs.filter(j => j.status === 'failed' || j.status === 'timeout').length;
                
                document.getElementById('summary').innerHTML = `
                    <span>Total: <b>${data.total_jobs}</b></span>
                    <span class="running">Running: <b>${data.running}</b></span>
                    <span class="completed">Completed: <b>${data.completed}</b></span>
                    <span class="completed">Success: <b>${successCount}</b></span>
                    <span class="failed">Failed: <b>${failedCount}</b></span>
                `;
                
                let html = '';
                for (let job of data.jobs) {
                    let statusClass = getStatusClass(job.status);
                    let logs = (job.messages || []).slice(-3).map(m => 
                        `<div>[${m.t}] ${m.msg}</div>`).join('');
                    
                    html += `<tr class="status-${statusClass}">
                        <td>${job.job_id}</td>
                        <td>${job.name || '-'}</td>
                        <td class="${statusClass}">${job.status || 'unknown'}</td>
                        <td>${formatTime(job.elapsed)}</td>
                        <td>${job.num_registered || 0}</td>
                        <td>${job.num_points3d || 0}</td>
                        <td><div class="log-box">${logs || '-'}</div></td>
                    </tr>`;
                }
                document.getElementById('jobsTable').innerHTML = html || 
                    '<tr><td colspan="7">No jobs yet</td></tr>';
            });
        }
        
        updateDashboard();
        setInterval(updateDashboard, 3000);
    </script>
</body>
</html>'''
        
        dashboard_file = self.dashboard_dir / '_dashboard.html'
        with open(dashboard_file, 'w') as f:
            f.write(html)
        print(f"Dashboard created: {dashboard_file}")
        print(f"  TIP: For best results, run in dashboard folder: python -m http.server 8000")
        print(f"       Then open: http://localhost:8000/_dashboard.html")
    
    def wait_all(self, timeout: float = 0, poll_interval: float = 5) -> Dict[str, dict]:
        """Wait for all jobs to complete, updating dashboard periodically."""
        import time as time_module
        
        print(f"\n{'='*60}")
        print(f"Waiting for {len(self.jobs)} mapping jobs...")
        if timeout > 0:
            print(f"Timeout: {timeout}s per job")
        print(f"Dashboard: {self.dashboard_dir / '_dashboard.html'}")
        print(f"{'='*60}")
        
        while True:
            all_done = True
            for job_id in list(self.jobs.keys()):
                result = self.check_job(job_id, timeout=timeout)
                if result is None:
                    all_done = False
            
            # Update dashboard
            self._update_dashboard_status()
            
            if all_done:
                break
            
            # Print brief status
            running = [j for j in self.jobs.values() if not j.finished]
            completed = [j for j in self.jobs.values() if j.finished]
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Running: {len(running)}, Completed: {len(completed)}")
            
            time_module.sleep(poll_interval)
        
        print(f"\n  All {len(self.jobs)} jobs completed!")
        
        # Stop background updater
        self._stop_background_updater()
        
        # Final dashboard update
        self._update_dashboard_status()
        
        # Return all results
        return {job_id: job.result for job_id, job in self.jobs.items()}
    
    def print_summary(self):
        """Print final summary of all jobs."""
        success = sum(1 for j in self.jobs.values() if j.result.get('success'))
        failed = len(self.jobs) - success
        
        print(f"\n{'='*60}")
        print(f"MAPPING SUMMARY: {success} success, {failed} failed")
        print(f"{'='*60}")
        
        for job_id, job in self.jobs.items():
            r = job.result
            status = "OK" if r.get('success') else "FAIL"
            print(f"  [{job_id}] {job.name}: {status} - "
                  f"Reg: {r.get('num_registered', 0)}, "
                  f"Pts: {r.get('num_points3d', 0)}, "
                  f"Time: {r.get('elapsed', 0):.1f}s")
            if r.get('error'):
                print(f"           Error: {r['error']}")
