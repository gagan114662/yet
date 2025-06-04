#!/usr/bin/env python3
"""
LARGE FILE MANAGEMENT - Utilities for managing large backtest files with Git LFS
"""

import os
import subprocess
import json
from pathlib import Path

class LargeFileManager:
    """Manage large files in the algorithmic trading system"""
    
    def __init__(self, repo_path=None):
        if repo_path is None:
            repo_path = "/mnt/VANDAN_DISK/gagan_stuff/again and again"
        self.repo_path = Path(repo_path)
        
    def find_large_files(self, size_mb=10):
        """Find files larger than specified size"""
        print(f"🔍 Finding files larger than {size_mb}MB...")
        
        large_files = []
        size_bytes = size_mb * 1024 * 1024
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            for file in files:
                file_path = Path(root) / file
                try:
                    if file_path.stat().st_size > size_bytes:
                        size_mb_actual = file_path.stat().st_size / (1024 * 1024)
                        relative_path = file_path.relative_to(self.repo_path)
                        large_files.append({
                            'path': str(relative_path),
                            'size_mb': round(size_mb_actual, 2)
                        })
                except (OSError, ValueError):
                    continue
        
        # Sort by size
        large_files.sort(key=lambda x: x['size_mb'], reverse=True)
        
        print(f"📊 Found {len(large_files)} files larger than {size_mb}MB:")
        for file_info in large_files[:10]:  # Show top 10
            print(f"   {file_info['size_mb']}MB - {file_info['path']}")
        
        if len(large_files) > 10:
            print(f"   ... and {len(large_files) - 10} more")
        
        return large_files
    
    def migrate_to_lfs(self, pattern):
        """Migrate existing files to LFS tracking"""
        print(f"📦 Migrating {pattern} to Git LFS...")
        
        try:
            # Track the pattern with LFS
            subprocess.run([
                'git', 'lfs', 'track', pattern
            ], cwd=self.repo_path, check=True)
            
            # Migrate existing files
            subprocess.run([
                'git', 'lfs', 'migrate', 'import', 
                '--include', pattern,
                '--everything'
            ], cwd=self.repo_path, check=True)
            
            print(f"✅ Successfully migrated {pattern} to LFS")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Migration failed: {e}")
    
    def check_lfs_status(self):
        """Check Git LFS status"""
        print(f"📊 Git LFS Status:")
        
        try:
            # Check LFS tracking
            result = subprocess.run([
                'git', 'lfs', 'track'
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("📋 Currently tracked patterns:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
            
            # Check LFS files
            result = subprocess.run([
                'git', 'lfs', 'ls-files'
            ], cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                lfs_files = result.stdout.strip().split('\n')
                lfs_files = [f for f in lfs_files if f.strip()]
                print(f"\n📦 LFS tracked files: {len(lfs_files)}")
                
                if lfs_files:
                    print("📋 Recent LFS files:")
                    for file in lfs_files[:5]:
                        print(f"   {file}")
                    if len(lfs_files) > 5:
                        print(f"   ... and {len(lfs_files) - 5} more")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error checking LFS status: {e}")
    
    def cleanup_old_results(self, days=30):
        """Remove old backtest results to save space"""
        print(f"🧹 Cleaning up backtest results older than {days} days...")
        
        import time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        patterns = [
            "**/backtests/**/*.json",
            "**/results/**/*.json", 
            "**/*RESULTS*.json"
        ]
        
        removed_count = 0
        removed_size = 0
        
        for pattern in patterns:
            for file_path in self.repo_path.glob(pattern):
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        removed_count += 1
                        removed_size += size
                        print(f"   Removed: {file_path.relative_to(self.repo_path)}")
                except (OSError, ValueError):
                    continue
        
        removed_size_mb = removed_size / (1024 * 1024)
        print(f"✅ Removed {removed_count} files, freed {removed_size_mb:.1f}MB")
    
    def archive_results(self, archive_path=None):
        """Archive old backtest results to external storage"""
        if archive_path is None:
            archive_path = self.repo_path / "archived_results"
        
        archive_path = Path(archive_path)
        archive_path.mkdir(exist_ok=True)
        
        print(f"📦 Archiving backtest results to: {archive_path}")
        
        # Find backtest result files
        patterns = [
            "**/backtests/**/*.json",
            "**/*RESULTS*.json"
        ]
        
        archived_count = 0
        
        for pattern in patterns:
            for file_path in self.repo_path.glob(pattern):
                if 'archived_results' not in str(file_path):
                    relative_path = file_path.relative_to(self.repo_path)
                    archive_file = archive_path / relative_path
                    
                    # Create directory structure
                    archive_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move file
                    file_path.rename(archive_file)
                    archived_count += 1
                    print(f"   Archived: {relative_path}")
        
        print(f"✅ Archived {archived_count} files")
    
    def setup_automated_cleanup(self):
        """Set up automated cleanup for large files"""
        print("⚙️ Setting up automated cleanup...")
        
        # Create cleanup script
        cleanup_script = self.repo_path / "scripts" / "cleanup_large_files.sh"
        cleanup_script.parent.mkdir(exist_ok=True)
        
        script_content = """#!/bin/bash
# Automated cleanup for large backtest files

echo "🧹 Running automated large file cleanup..."

# Remove files older than 30 days
find . -name "*RESULTS*.json" -mtime +30 -delete
find . -path "*/backtests/*" -name "*.json" -mtime +30 -delete
find . -name "*.log" -size +100M -mtime +7 -delete

# Compress old evolution checkpoints
find . -name "*checkpoint*.json" -mtime +7 -exec gzip {} \;

echo "✅ Cleanup completed"
"""
        
        with open(cleanup_script, 'w') as f:
            f.write(script_content)
        
        cleanup_script.chmod(0o755)
        print(f"✅ Created cleanup script: {cleanup_script}")
        
        # Create .gitignore entries for temporary files
        gitignore_additions = """
# Large temporary files
*.tmp
*.temp
**/temp/**
**/tmp/**

# Compressed archives
*.gz
*.tar.gz
*.zip

# Old log files
*.log.old
*.log.[0-9]*
"""
        
        gitignore_path = self.repo_path / ".gitignore"
        with open(gitignore_path, 'a') as f:
            f.write(gitignore_additions)
        
        print("✅ Updated .gitignore for temporary files")

def main():
    """Run large file management operations"""
    
    print("🗃️ LARGE FILE MANAGEMENT TOOL")
    print("=" * 50)
    
    manager = LargeFileManager()
    
    print("\n1. 🔍 CHECKING FOR LARGE FILES")
    large_files = manager.find_large_files(50)  # Files > 50MB
    
    print("\n2. 📊 GIT LFS STATUS")
    manager.check_lfs_status()
    
    print("\n3. ⚙️ SETUP AUTOMATED CLEANUP")
    manager.setup_automated_cleanup()
    
    if large_files:
        print(f"\n⚠️ RECOMMENDATIONS:")
        print(f"   • {len([f for f in large_files if f['size_mb'] > 100])} files > 100MB should use LFS")
        print(f"   • Consider archiving old results with: manager.archive_results()")
        print(f"   • Run cleanup with: manager.cleanup_old_results(days=30)")
        print(f"   • Large files will automatically use LFS for future commits")
    else:
        print(f"\n✅ No files larger than 50MB found")
    
    print(f"\n💡 FUTURE LARGE FILES:")
    print(f"   • Will automatically be tracked by Git LFS")
    print(f"   • No manual intervention needed")
    print(f"   • Repository will stay lightweight")

if __name__ == "__main__":
    main()