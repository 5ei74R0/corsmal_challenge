## When you start
1. install dependencies  
    use poetry (recommended)
    ```sh
    $ poetry install
    ```
    or
    ```sh
    $ conda env update --file env.yml
    ```

## If you want to update new dependencies
1. add by **`poetry add` to update pyproject.toml**
2. regenerate `env.yml`
    ```sh
    conda env export > `env.yml`
    ```
    and remove `name:` & `prefix:` from `env.yml`
