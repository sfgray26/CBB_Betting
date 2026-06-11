#!/usr/bin/env python3
"""
Agent Orchestrator for CBB Edge Multi-Agent System

Automates task delegation across Claude, Codex, and Gemini based on task characteristics.

Usage:
    python scripts/agent_orchestrator.py --task "Implement BDL #3 injury overlay" --parallel
    python scripts/agent_orchestrator.py --task "Fix bug in fantasy.py" --agent claude
    python scripts/agent_orchestrator.py --batch tasks.json

Requirements:
    - claude, codex, gemini CLIs installed and authenticated
    - Git repo with clean working directory
    - tasks.json for batch mode
"""

import argparse
import json
import subprocess
import sys
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from enum import Enum


class AgentType(Enum):
    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"


class TaskType(Enum):
    COMPLEX_BUG = "complex_bug"
    SINGLE_FEATURE = "single_feature"
    MULTI_FILE_REFACTOR = "multi_file_refactor"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    DATABASE_MIGRATION = "database_migration"
    API_INTEGRATION = "api_integration"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


@dataclass
class Task:
    id: str
    description: str
    task_type: str
    files_affected: List[str]
    priority: int
    estimated_minutes: int
    blocked_by: Optional[List[str]] = None
    context_files: Optional[List[str]] = None
    agent: Optional[str] = None  # Optional forced agent


@dataclass
class AgentAssignment:
    task: Task
    agent: AgentType
    rationale: str
    prompt: str
    branch_name: str


class AgentRouter:
    """Routes tasks to appropriate agents based on task characteristics."""
    
    def route(self, task: Task) -> AgentType:
        """Determine best agent for a task using decision tree."""
        
        # If agent is forced, use it
        if task.agent:
            return AgentType(task.agent)
        
        # P0/P1 bugs -> Claude (deep analysis)
        if task.priority <= 1:
            return AgentType.CLAUDE
        
        # Multi-file changes -> Claude (coordination)
        if len(task.files_affected) > 3:
            return AgentType.CLAUDE
        
        # Tests or migrations -> Codex (fast implementation)
        if task.task_type in [TaskType.TEST_GENERATION, TaskType.DATABASE_MIGRATION]:
            return AgentType.CODEX
        
        # Review or docs -> Gemini (thorough, structured)
        if task.task_type in [TaskType.CODE_REVIEW, TaskType.DOCUMENTATION]:
            return AgentType.GEMINI
        
        # Complex architecture -> Claude
        if task.task_type in [TaskType.COMPLEX_BUG, TaskType.MULTI_FILE_REFACTOR, 
                              TaskType.API_INTEGRATION, TaskType.PERFORMANCE_OPTIMIZATION]:
            return AgentType.CLAUDE
        
        # Default: Codex for single features
        return AgentType.CODEX
    
    def get_rationale(self, task: Task, agent: AgentType) -> str:
        """Explain why this agent was chosen."""
        rationales = {
            AgentType.CLAUDE: "Complex task requiring deep reasoning and multi-file coordination",
            AgentType.CODEX: "Straightforward implementation - optimize for speed",
            AgentType.GEMINI: "Review/documentation task requiring thoroughness",
        }
        return rationales[agent]


class PromptBuilder:
    """Builds optimized prompts for each agent type."""
    
    def build(self, task: Task, agent: AgentType) -> str:
        """Generate agent-specific prompt."""
        
        base_context = self._load_context_files(task.context_files or [])
        
        builders = {
            AgentType.CLAUDE: self._build_claude_prompt,
            AgentType.CODEX: self._build_codex_prompt,
            AgentType.GEMINI: self._build_gemini_prompt,
        }
        
        return builders[agent](task, base_context)
    
    def _load_context_files(self, files: List[str]) -> str:
        """Load content from context files."""
        context = []
        for file in files:
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    context.append(f"--- {file} ---\n{content[:2000]}...")  # Truncate long files
            except Exception as e:
                context.append(f"--- {file} ---\nError loading: {e}")
        return "\n\n".join(context)
    
    def _build_claude_prompt(self, task: Task, context: str) -> str:
        return f"""Read HERMES.md and SKILL.md for project context.

Task: {task.description}
Priority: P{task.priority}
Files affected: {', '.join(task.files_affected)}

Context:
{context}

Requirements:
- Provide exact file paths and line numbers
- Include comprehensive error handling
- Add regression tests for any bug fixes
- Follow patterns in cbb-edge-workflow skill
- Run full pytest suite before declaring done
- Update SKILL.md if you discover new patterns

Expected outcome: Working implementation with tests, no regressions.
"""
    
    def _build_codex_prompt(self, task: Task, context: str) -> str:
        return f"""Implement the following feature/fix:

Task: {task.description}
Files to modify: {', '.join(task.files_affected)}

Context from codebase:
{context}

Do:
- Implement the feature efficiently
- Add type hints
- Write 3-5 unit tests in test_{task.id}.py
- Run pytest and ensure all tests pass
- Follow existing code patterns

Don't:
- Modify unrelated files
- Skip error handling
- Leave TODOs or FIXMEs
- Break existing tests

Expected: Working implementation + passing tests.
"""
    
    def _build_gemini_prompt(self, task: Task, context: str) -> str:
        return f"""Review and improve the following code:

Task: {task.description}
Files: {', '.join(task.files_affected)}

Code context:
{context}

Your task:
1. Review for:
   - Python best practices (PEP 8)
   - SQL injection risks
   - Missing docstrings
   - Type safety issues
   - Performance concerns
   - Security vulnerabilities

2. Generate:
   - API documentation (if applicable)
   - README updates
   - Inline comments for complex logic
   - Usage examples

3. Verify:
   - All public functions have docstrings
   - Complex algorithms have comments
   - API contracts are documented

Output: Review report + documentation files.
"""


class BranchNamer:
    """Generates consistent branch names."""
    
    def generate(self, task: Task, agent: AgentType) -> str:
        """Create branch name from task and agent."""
        timestamp = datetime.now().strftime("%Y%m%d")
        task_slug = re.sub(r'[^\w]', '-', task.description.lower())[:40]
        return f"agent/{agent.value}/{task_slug}-{timestamp}"


class TaskExecutor:
    """Executes tasks using appropriate agent CLI."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.results = []
    
    def execute(self, assignment: AgentAssignment) -> Dict:
        """Execute a task assignment."""
        
        print(f"\n{'='*60}")
        print(f"Executing: {assignment.task.id}")
        print(f"Agent: {assignment.agent.value}")
        print(f"Branch: {assignment.branch_name}")
        print(f"Rationale: {assignment.rationale}")
        print(f"{'='*60}\n")
        
        if self.dry_run:
            print("[DRY RUN] Would execute:")
            print(f"  git checkout -b {assignment.branch_name}")
            print(f"  {assignment.agent.value} -p \"<prompt>\" --permission-mode bypassPermissions")
            return {"status": "dry_run", "task": assignment.task.id}
        
        # Create branch
        try:
            subprocess.run(
                ["git", "checkout", "-b", assignment.branch_name],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✓ Created branch: {assignment.branch_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create branch: {e}")
            return {"status": "failed", "error": str(e)}
        
        # Build command
        cmd = [
            assignment.agent.value,
            "-p", assignment.prompt,
            "--permission-mode", "bypassPermissions"
        ]
        
        print(f"Running: {' '.join(cmd[:3])}...")  # Don't print full prompt
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=assignment.task.estimated_minutes * 60
            )
            
            status = "success" if result.returncode == 0 else "failed"
            
            result_data = {
                "status": status,
                "task": assignment.task.id,
                "agent": assignment.agent.value,
                "branch": assignment.branch_name,
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:],  # Last 2000 chars
                "stderr": result.stderr[-1000:],  # Last 1000 chars
            }
            
            self.results.append(result_data)
            
            if status == "success":
                print(f"✓ Task {assignment.task.id} completed")
            else:
                print(f"✗ Task {assignment.task.id} failed")
                print(f"  Error: {result.stderr[-500:]}")
            
            return result_data
            
        except subprocess.TimeoutExpired:
            print(f"✗ Task {assignment.task.id} timed out")
            return {"status": "timeout", "task": assignment.task.id}
        except Exception as e:
            print(f"✗ Task {assignment.task.id} error: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_report(self) -> str:
        """Generate execution report."""
        lines = ["\n" + "="*60, "EXECUTION REPORT", "="*60]
        
        for result in self.results:
            status_icon = "✓" if result["status"] == "success" else "✗"
            lines.append(f"\n{status_icon} {result['task']} ({result.get('agent', 'unknown')})")
            lines.append(f"   Status: {result['status']}")
            lines.append(f"   Branch: {result.get('branch', 'N/A')}")
            
            if result["status"] != "success" and "stderr" in result:
                lines.append(f"   Error preview: {result['stderr'][:200]}")
        
        success_count = sum(1 for r in self.results if r["status"] == "success")
        lines.append(f"\n{'='*60}")
        lines.append(f"Summary: {success_count}/{len(self.results)} tasks successful")
        lines.append(f"{'='*60}\n")
        
        return "\n".join(lines)


class Orchestrator:
    """Main orchestration logic."""
    
    def __init__(self, dry_run: bool = False):
        self.router = AgentRouter()
        self.prompt_builder = PromptBuilder()
        self.branch_namer = BranchNamer()
        self.executor = TaskExecutor(dry_run=dry_run)
        self.dry_run = dry_run
    
    def orchestrate_single(self, description: str, agent: Optional[str] = None,
                          priority: int = 3, files: Optional[List[str]] = None) -> Dict:
        """Orchestrate a single task."""
        
        task_id = f"task_{datetime.now().strftime('%H%M%S')}"
        
        # Infer task type from description
        task_type = self._infer_task_type(description)
        
        task = Task(
            id=task_id,
            description=description,
            task_type=task_type.value,
            files_affected=files or [],
            priority=priority,
            estimated_minutes=30,
            blocked_by=None,
            context_files=["HERMES.md", ".agent-handoffs/SKILL.md"],
            agent=agent
        )
        
        # Route to agent
        selected_agent = self.router.route(task)
        
        rationale = self.router.get_rationale(task, selected_agent)
        prompt = self.prompt_builder.build(task, selected_agent)
        branch = self.branch_namer.generate(task, selected_agent)
        
        assignment = AgentAssignment(
            task=task,
            agent=selected_agent,
            rationale=rationale,
            prompt=prompt,
            branch_name=branch
        )
        
        return self.executor.execute(assignment)
    
    def orchestrate_parallel(self, tasks: List[Task]) -> List[Dict]:
        """Orchestrate multiple tasks in parallel."""
        
        print(f"\n🚀 Orchestrating {len(tasks)} tasks in parallel\n")
        
        assignments = []
        for task in tasks:
            agent = self.router.route(task)
            assignment = AgentAssignment(
                task=task,
                agent=agent,
                rationale=self.router.get_rationale(task, agent),
                prompt=self.prompt_builder.build(task, agent),
                branch_name=self.branch_namer.generate(task, agent)
            )
            assignments.append(assignment)
        
        # Display plan
        print("Execution Plan:")
        print("-" * 60)
        for i, assignment in enumerate(assignments, 1):
            print(f"{i}. {assignment.task.id}")
            print(f"   Agent: {assignment.agent.value}")
            print(f"   Branch: {assignment.branch_name}")
            print(f"   Why: {assignment.rationale}")
            print()
        
        if self.dry_run:
            print("[DRY RUN] Parallel execution would start now")
            return []
        
        # Note: True parallel execution would require threading/multiprocessing
        # For now, we execute sequentially but the plan shows the parallel intent
        results = []
        for assignment in assignments:
            result = self.executor.execute(assignment)
            results.append(result)
        
        print(self.executor.generate_report())
        return results
    
    def _infer_task_type(self, description: str) -> TaskType:
        """Infer task type from description."""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ["bug", "fix", "error", "crash"]):
            return TaskType.COMPLEX_BUG
        elif any(word in desc_lower for word in ["test", "testing"]):
            return TaskType.TEST_GENERATION
        elif any(word in desc_lower for word in ["review", "audit"]):
            return TaskType.CODE_REVIEW
        elif any(word in desc_lower for word in ["doc", "documentation", "readme"]):
            return TaskType.DOCUMENTATION
        elif any(word in desc_lower for word in ["migration", "schema", "table"]):
            return TaskType.DATABASE_MIGRATION
        elif any(word in desc_lower for word in ["api", "integration", "endpoint"]):
            return TaskType.API_INTEGRATION
        elif any(word in desc_lower for word in ["performance", "optimize", "slow"]):
            return TaskType.PERFORMANCE_OPTIMIZATION
        elif any(word in desc_lower for word in ["refactor", "restructure"]):
            return TaskType.MULTI_FILE_REFACTOR
        else:
            return TaskType.SINGLE_FEATURE


def main():
    parser = argparse.ArgumentParser(
        description="Agent Orchestrator for CBB Edge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single task with auto-routing
    python agent_orchestrator.py --task "Fix bug in matchup scoring"
    
    # Single task with specific agent
    python agent_orchestrator.py --task "Write tests" --agent codex
    
    # Parallel execution of multiple tasks
    python agent_orchestrator.py --task "Feature A" --task "Feature B" --parallel
    
    # Batch mode from JSON
    python agent_orchestrator.py --batch tasks.json
    
    # Dry run (see what would happen)
    python agent_orchestrator.py --task "Some task" --dry-run
        """
    )
    
    parser.add_argument("--task", action="append", help="Task description (can specify multiple)")
    parser.add_argument("--agent", choices=["claude", "codex", "gemini"], 
                       help="Force specific agent (auto-selected if omitted)")
    parser.add_argument("--priority", type=int, default=3, 
                       help="Task priority 1-5 (1=highest, default=3)")
    parser.add_argument("--files", nargs="+", help="Files affected by task")
    parser.add_argument("--parallel", action="store_true", 
                       help="Run tasks in parallel")
    parser.add_argument("--batch", help="JSON file with task batch")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be executed without running")
    
    args = parser.parse_args()
    
    orchestrator = Orchestrator(dry_run=args.dry_run)
    
    if args.batch:
        # Batch mode
        with open(args.batch, 'r') as f:
            batch = json.load(f)
        tasks = [Task(**t) for t in batch["tasks"]]
        orchestrator.orchestrate_parallel(tasks)
    
    elif args.task:
        if len(args.task) == 1 and not args.parallel:
            # Single task
            result = orchestrator.orchestrate_single(
                description=args.task[0],
                agent=args.agent,
                priority=args.priority,
                files=args.files
            )
            print(f"\nResult: {result['status']}")
        else:
            # Multiple tasks (parallel or sequential)
            tasks = []
            for i, desc in enumerate(args.task):
                tasks.append(Task(
                    id=f"task_{i+1}",
                    description=desc,
                    task_type=orchestrator._infer_task_type(desc).value,
                    files_affected=args.files or [],
                    priority=args.priority,
                    estimated_minutes=30,
                    blocked_by=None,
                    context_files=None,
                    agent=args.agent
                ))
            orchestrator.orchestrate_parallel(tasks)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
