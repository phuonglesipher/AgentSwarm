param(
    [switch]$Once,
    [switch]$Poll,
    [int]$IntervalSeconds = 60,
    [switch]$DryRun,
    [string]$CodexProfile = "ai-gateway",
    [string]$CodexModel = "gpt-5.5",
    [string]$ReasoningEffort = "xhigh",
    [string]$CodexSandbox = "danger-full-access",
    [int]$CodexTimeoutSeconds = 1800,
    [int]$MaxTasks = 0,
    [string]$Repo = "sipherxyz/s2",
    [string]$ProjectOwner = "sipherxyz",
    [int]$ProjectNumber = 5,
    [string]$Label = "AgentTask"
)

$ErrorActionPreference = "Stop"

$AgentSwarmRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $AgentSwarmRoot

function Import-DotEnvFile {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            return
        }
        if ($line -notmatch '^(?:export\s+)?([^#=\s]+)\s*=\s*(.*)$') {
            return
        }

        $name = $matches[1].Trim()
        $value = $matches[2].Trim()

        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        } elseif ($value.Contains("#")) {
            $value = ($value -replace '\s+#.*$', '').Trim()
        }

        if ($name) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

Import-DotEnvFile (Join-Path $RepoRoot ".env")
Import-DotEnvFile (Join-Path $RepoRoot ".env.local")
Import-DotEnvFile (Join-Path $AgentSwarmRoot ".env")
Import-DotEnvFile (Join-Path $AgentSwarmRoot ".env.local")

if (-not $PSBoundParameters.ContainsKey("Repo") -and $env:AGENT_TASK_REPO) {
    $Repo = $env:AGENT_TASK_REPO
}
if (-not $PSBoundParameters.ContainsKey("Label") -and $env:AGENT_TASK_LABEL) {
    $Label = $env:AGENT_TASK_LABEL
}
if (-not $PSBoundParameters.ContainsKey("ProjectOwner") -and $env:AGENT_TASK_PROJECT_OWNER) {
    $ProjectOwner = $env:AGENT_TASK_PROJECT_OWNER
}
if (-not $PSBoundParameters.ContainsKey("ProjectNumber") -and $env:AGENT_TASK_PROJECT_NUMBER) {
    $ProjectNumber = [int]$env:AGENT_TASK_PROJECT_NUMBER
}
if (-not $PSBoundParameters.ContainsKey("CodexProfile")) {
    if ($env:AGENT_TASK_CODEX_PROFILE) {
        $CodexProfile = $env:AGENT_TASK_CODEX_PROFILE
    } elseif ($env:CODEX_PROFILE) {
        $CodexProfile = $env:CODEX_PROFILE
    }
}
if (-not $PSBoundParameters.ContainsKey("CodexModel")) {
    if ($env:AGENT_TASK_CODEX_MODEL) {
        $CodexModel = $env:AGENT_TASK_CODEX_MODEL
    } elseif ($env:CODEX_MODEL) {
        $CodexModel = $env:CODEX_MODEL
    }
}
if (-not $PSBoundParameters.ContainsKey("ReasoningEffort")) {
    if ($env:AGENT_TASK_REASONING_EFFORT) {
        $ReasoningEffort = $env:AGENT_TASK_REASONING_EFFORT
    } elseif ($env:CODEX_REASONING_EFFORT) {
        $ReasoningEffort = $env:CODEX_REASONING_EFFORT
    }
}
if (-not $PSBoundParameters.ContainsKey("CodexSandbox")) {
    if ($env:AGENT_TASK_CODEX_SANDBOX) {
        $CodexSandbox = $env:AGENT_TASK_CODEX_SANDBOX
    } elseif ($env:CODEX_SANDBOX) {
        $CodexSandbox = $env:CODEX_SANDBOX
    }
}
if (-not $PSBoundParameters.ContainsKey("CodexTimeoutSeconds") -and $env:AGENT_TASK_CODEX_TIMEOUT_SECONDS) {
    $CodexTimeoutSeconds = [int]$env:AGENT_TASK_CODEX_TIMEOUT_SECONDS
}

$modeArgs = @()
if ($Poll) {
    $modeArgs += "--poll"
} else {
    $modeArgs += "--once"
}
if ($DryRun) {
    $modeArgs += "--dry-run"
}

$arguments = @(
    (Join-Path $AgentSwarmRoot "main.py"),
    "--mode", "agent-task",
    "--host-root", $RepoRoot,
    "--interval-seconds", "$IntervalSeconds",
    "--agent-task-repo", $Repo,
    "--agent-task-label", $Label,
    "--project-owner", $ProjectOwner,
    "--project-number", "$ProjectNumber",
    "--codex-profile", $CodexProfile,
    "--codex-model", $CodexModel,
    "--reasoning-effort", $ReasoningEffort,
    "--codex-sandbox", $CodexSandbox,
    "--codex-timeout-seconds", "$CodexTimeoutSeconds",
    "--max-tasks", "$MaxTasks"
) + $modeArgs

Write-Host "AgentSwarm AgentTask runner"
Write-Host "Repo root: $RepoRoot"
Write-Host "Codex: profile=$CodexProfile model=$CodexModel effort=$ReasoningEffort"

& python @arguments
exit $LASTEXITCODE
