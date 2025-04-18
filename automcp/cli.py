import argparse
import sys
import subprocess
import yaml
from pathlib import Path

# Determine the location of the templates relative to this file
_CLI_DIR = Path(__file__).parent
_TEMPLATE_FILE = _CLI_DIR / "cli_templates/run_mcp.py.template"
_CONFIG_FILE = _CLI_DIR / "cli_templates/framework_config.yaml"
_PYPROJECT_TEMPLATE_FILE = _CLI_DIR / "cli_templates/pyproject.toml.template"


def create_mcp_server_file(directory: Path, framework: str) -> None:
    """Create a run_mcp.py file in the specified directory using a template."""
    # Check if the unified template exists
    if not _TEMPLATE_FILE.exists():
        raise ValueError(f"Unified template file not found at: {_TEMPLATE_FILE}")
    
    # Load the configuration file
    if not _CONFIG_FILE.exists():
        raise ValueError(f"Configuration file not found at: {_CONFIG_FILE}")
    
    try:
        with open(_CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error loading configuration file: {e}")
    
    # Check if the specified framework exists in the configuration
    if framework not in config.get("frameworks", {}):
        raise ValueError(f"Framework '{framework}' not found in configuration")
    
    # Get the framework-specific configuration
    framework_config = config["frameworks"][framework]
    
    # Load the template
    with open(_TEMPLATE_FILE, "r") as f:
        template_content = f.read()
    
    # Replace placeholders with framework-specific values
    content = template_content
    
    # Add the framework name from the key
    content = content.replace("{{framework}}", framework)
    
    # Extract variable name from adapter definition
    adapter_variable_name = None
    adapter_def = framework_config.get("adapter_definition", "")
    first_line = adapter_def.strip().split("\n")[0].strip()
    if "=" in first_line:
        adapter_variable_name = first_line.split("=")[0].strip()
    
    # Default to framework if we couldn't extract it
    if not adapter_variable_name:
        adapter_variable_name = f"mcp_{framework}"
    
    # Replace all placeholders
    for key, value in framework_config.items():
        placeholder = f"{{{{{key}}}}}"
        content = content.replace(placeholder, value)
    
    # Replace adapter_variable_name placeholder
    content = content.replace("{{adapter_variable_name}}", adapter_variable_name)
    
    # Write the file
    file_path = directory / "run_mcp.py"
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"Created {file_path} from unified template for {framework} framework.")


def init_command(args) -> None:
    """Create new MCP server files in the current directory."""
    current_dir = Path.cwd()

    # Create run_mcp.py
    try:
        create_mcp_server_file(current_dir, args.framework)
    except ValueError as e:
        print(f"Error creating server file: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error writing server file: {e}", file=sys.stderr)
        sys.exit(1)

    # Create pyproject.toml if it doesn't exist
    pyproject_path = current_dir / "pyproject.toml"
    if not pyproject_path.exists():
        if not _PYPROJECT_TEMPLATE_FILE.exists():
            print(f"Warning: pyproject.toml template not found at {_PYPROJECT_TEMPLATE_FILE}", file=sys.stderr)
        else:
            try:
                with open(_PYPROJECT_TEMPLATE_FILE, "r") as f_template, open(pyproject_path, "w") as f_dest:
                    f_dest.write(f_template.read())
                print(f"Created {pyproject_path}")
            except IOError as e:
                print(f"Error writing {pyproject_path}: {e}", file=sys.stderr)
    else:
        print(f"{pyproject_path} already exists, skipping creation.")

    print("\nSetup complete! Next steps:")
    print(f"1. Edit {current_dir / 'run_mcp.py'} to import and configure your {args.framework} agent/crew/graph")
    print(f"2. Edit {pyproject_path} to add framework-specific dependencies (e.g., crewai, langgraph)")
    print("3. Add a .env file with necessary environment variables")
    print("4. Run your MCP server using one of these commands:")
    print("   - automcp serve         # For STDIO transport (default)")
    print("   - automcp serve -t sse     # For SSE transport")
    print("   - uv run serve_stdio           # Using the script entry point (if using uv)")
    print("   - uv run serve_sse             # Using the script entry point (if using uv)")


def serve_command(args) -> None:
    """Run the AutoMCP server."""
    print(f"Running AutoMCP server with {args.transport} transport")
    current_dir = Path.cwd()
    
    automcp_file = current_dir / "run_mcp.py"
    if not automcp_file.exists():
        raise ValueError("run_mcp.py not found in current directory")
    
    venv_dir = current_dir / ".venv"
    if not venv_dir.exists():
        pyproject_file = current_dir / "pyproject.toml"
        if not pyproject_file.exists():
            raise ValueError("No venv found and no pyproject.toml to create one")
        
        try:
            subprocess.run(["uv", "venv"], check=True)
            # TODO: remove --no-cache once testing phase is done
            subprocess.run(["uv", "sync", "--no-cache"], check=True)
            
            # This is to install the automcp package
            requirements_file = current_dir / "requirements.txt"
            if requirements_file.exists():
                subprocess.run(["uv", "add", "-r", str(requirements_file)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error setting up environment: {e}", file=sys.stderr)
            sys.exit(1)
    
    try:
        if args.transport == "stdio":
            subprocess.run(["uv", "run", str(automcp_file)], check=True)
        elif args.transport == "sse":
            subprocess.run(["uv", "run", str(automcp_file), "sse"], check=True)
        else:
            raise ValueError(f"Invalid transport: {args.transport}")
    except subprocess.CalledProcessError as e:
        print(f"Error running AutoMCP server: {e}", file=sys.stderr)
        sys.exit(1)


def load_available_frameworks():
    """Load available frameworks from the configuration file."""
    try:
        with open(_CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)
        return list(config.get("frameworks", {}).keys())
    except Exception:
        # Fallback to hardcoded list if config file can't be loaded
        return [
            "crewai","langgraph", "pydantic",
            "llamaindex", "openai",  "mcp_agent", 
        ]


def main():
    parser = argparse.ArgumentParser(description="AutoMCP - Convert agents to MCP servers")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # Get available frameworks from the config file
    available_frameworks = load_available_frameworks()
    
    # init command
    init_parser = subparsers.add_parser("init", help="Create a new MCP server configuration")
    init_parser.add_argument(
        "-f",
        "--framework",
        choices=available_frameworks,
        required=True,
        help=f"Agent framework to use (choices: {', '.join(available_frameworks)})"
    )
    init_parser.set_defaults(func=init_command)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run the AutoMCP server")
    serve_parser.add_argument(
        "-t",
        "--transport",
        nargs="?",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport to use (stdio or sse, defaults to stdio)"
    )
    serve_parser.set_defaults(func=serve_command)

    # Parse args and call the appropriate function
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()