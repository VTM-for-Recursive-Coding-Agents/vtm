{
  description = "Nix flake for VTM development and packaging";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      eachSystem = f:
        nixpkgs.lib.genAttrs systems (system:
          f {
            inherit system;
            pkgs = import nixpkgs { inherit system; };
          });
    in
    {
      packages = eachSystem ({ pkgs, ... }:
        let
          python = pkgs.python312;
          pythonPackages = pkgs.python312Packages;
          vtm = pythonPackages.buildPythonPackage {
            pname = "vtm";
            version = "0.1.0";
            pyproject = true;
            src = ./.;

            nativeBuildInputs = with pythonPackages; [
              hatchling
            ];

            propagatedBuildInputs = with pythonPackages; [
              pydantic
              tree-sitter
              tree-sitter-python
            ];

            pythonImportsCheck = [ "vtm" ];

            meta = with pkgs.lib; {
              description = "Verified Transactional Memory kernel for coding agents";
              homepage = "https://github.com/VTM-for-Recursive-Coding-Agents/vtm";
              license = licenses.mit;
              platforms = platforms.unix;
            };
          };
        in
        {
          default = vtm;
          vtm = vtm;
        });

      apps = eachSystem ({ pkgs, system, ... }:
        let
          vtm = self.packages.${system}.vtm;
        in
        {
          default = {
            type = "app";
            program = "${vtm}/bin/vtm-bench";
          };
          vtm-bench = {
            type = "app";
            program = "${vtm}/bin/vtm-bench";
          };
          vtm-prepare-swebench-lite = {
            type = "app";
            program = "${vtm}/bin/vtm-prepare-swebench-lite";
          };
        });

      devShells = eachSystem ({ pkgs, ... }:
        let
          python = pkgs.python312;
          pythonPackages = pkgs.python312Packages;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.git
              pkgs.docker
              pkgs.sqlite
              pkgs.ripgrep
              pythonPackages.mypy
              pythonPackages.pytest
              pythonPackages.ruff
              pythonPackages.pydantic
              pythonPackages.tree-sitter
              pythonPackages.tree-sitter-python
              pythonPackages.openai
              pythonPackages.datasets
            ];

            shellHook = ''
              export UV_PYTHON=${python}/bin/python3.12
              echo "VTM nix dev shell"
              echo "Run: uv sync --dev --all-extras"
            '';
          };
        });

      formatter = eachSystem ({ pkgs, ... }: pkgs.nixpkgs-fmt);
    };
}
