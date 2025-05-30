# Contributing

Thanks for considering contributing to NVIDIA Isaac-GR00T! Please read this document to learn the various ways you can contribute to this project and how to go about doing it.

## Bug reports and feature requests

### Did you find a bug?

First, do [a quick search](https://github.com/NVIDIA/Isaac-GR00T/issues) to see whether your issue has already been reported.
If your issue has already been reported, please comment on the existing issue.

Otherwise, open [a new GitHub issue](https://github.com/NVIDIA/Isaac-GR00T/issues/new).  Be sure to include a clear title
and description.  The description should include as much relevant information as possible.  The description should
explain how to reproduce the erroneous behavior as well as the behavior you expect to see.  Ideally you would include a
code sample or an executable test case demonstrating the expected behavior.

### Do you have a suggestion for an enhancement or new feature?

We use GitHub issues to track feature requests. Before you create a feature request:

* Make sure you have a clear idea of the enhancement you would like. If you have a vague idea, consider discussing
it first on a GitHub issue.
* Check the documentation to make sure your feature does not already exist.
* Do [a quick search](https://github.com/NVIDIA/Isaac-GR00T/issues) to see whether your feature has already been suggested.

When creating your request, please:

* Provide a clear title and description.
* Explain why the enhancement would be useful. It may be helpful to highlight the feature in other libraries.
* Include code examples to demonstrate how the enhancement would be used.

## Making a pull request

When you're ready to contribute code to address an open issue, please follow these guidelines to help us be able to review your pull request (PR) quickly.

1. **Initial setup** (only do this once)

    <details><summary>Expand details 👇</summary><br/>

    If you haven't already done so, please [fork](https://help.github.com/en/enterprise/2.13/user/articles/fork-a-repo) this repository on GitHub.

    Then clone your fork locally with

        git clone https://github.com/USERNAME/Isaac-GR00T.git

    or 

        git clone git@github.com:USERNAME/Isaac-GR00T.git

    At this point the local clone of your fork only knows that it came from *your* repo, github.com/USERNAME/Isaac-GR00T.git, but doesn't know anything the *main* repo, [https://github.com/NVIDIA/Isaac-GR00T.git](https://github.com/NVIDIA/Isaac-GR00T). You can see this by running

        git remote -v

    which will output something like this:

        origin https://github.com/USERNAME/Isaac-GR00T.git (fetch)
        origin https://github.com/USERNAME/Isaac-GR00T.git (push)

    This means that your local clone can only track changes from your fork, but not from the main repo, and so you won't be able to keep your fork up-to-date with the main repo over time. Therefore you'll need to add another "remote" to your clone that points to [https://github.com/NVIDIA/Isaac-GR00T.git](https://github.com/NVIDIA/Isaac-GR00T). To do this, run the following:

        git remote add upstream https://github.com/NVIDIA/Isaac-GR00T.git

    Now if you do `git remote -v` again, you'll see

        origin https://github.com/USERNAME/Isaac-GR00T.git (fetch)
        origin https://github.com/USERNAME/Isaac-GR00T.git (push)
        upstream https://github.com/NVIDIA/Isaac-GR00T.git (fetch)
        upstream https://github.com/NVIDIA/Isaac-GR00T.git (push)

    Finally, you'll need to create a Python 3 virtual environment suitable for working on this project. There a number of tools out there that making working with virtual environments easier.
    The most direct way is with the [`venv` module](https://docs.python.org/3.7/library/venv.html) in the standard library, but if you're new to Python or you don't already have a recent Python 3 version installed on your machine,
    we recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

    On Mac, for example, you can install Miniconda with [Homebrew](https://brew.sh/):

        brew install miniconda

    Then you can create and activate a new Python environment by running:

        conda create -n isaac-gr00t python=3.10
        conda activate isaac-gr00t

    Once your virtual environment is activated, you can install your local clone in "editable mode" with

        pip install -U pip setuptools wheel
        pip install -e .[dev]

    The "editable mode" comes from the `-e` argument to `pip`, and essential just creates a symbolic link from the site-packages directory of your virtual environment to the source code in your local clone. That way any changes you make will be immediately reflected in your virtual environment.

    </details>

2. **Ensure your fork is up-to-date**

    <details><summary>Expand details 👇</summary><br/>

    Once you've added an "upstream" remote pointing to [https://github.com/NVIDIA/Isaac-GR00T.git](https://github.com/NVIDIA/Isaac-GR00T), keeping your fork up-to-date is easy:

        git checkout main  # if not already on main
        git pull --rebase upstream main
        git push

    </details>

3. **Create a new branch to work on your fix or enhancement**

    <details><summary>Expand details 👇</summary><br/>

    Committing directly to the main branch of your fork is not recommended. It will be easier to keep your fork clean if you work on a separate branch for each contribution you intend to make.

    You can create a new branch with

        # replace BRANCH with whatever name you want to give it
        git checkout -b BRANCH
        git push -u origin BRANCH

    </details>

4. **Test your changes**

    <details><summary>Expand details 👇</summary><br/>

    Our continuous integration (CI) testing runs [a number of checks](https://github.com/NVIDIA/Isaac-GR00T/actions) for each pull request on [GitHub Actions](https://github.com/features/actions). You can run most of these tests locally, which is something you should do *before* opening a PR to help speed up the review process and make it easier for us.

    First, you should run [`isort`](https://github.com/PyCQA/isort) and [`black`](https://github.com/psf/black) to make sure you code is formatted consistently.
    Many IDEs support code formatters as plugins, so you may be able to setup isort and black to run automatically everytime you save.
    For example, [`black.vim`](https://github.com/psf/black/tree/master/plugin) will give you this functionality in Vim. But both `isort` and `black` are also easy to run directly from the command line.
    Just run this from the root of your clone:

        isort .
        black .

    Our CI also uses [`ruff`](https://github.com/astral-sh/ruff) to lint the code base and [`mypy`](http://mypy-lang.org/) for type-checking. You should run both of these next with

        ruff check . --fix

    We also strive to maintain high test coverage, so most contributions should include additions to [the unit tests](https://github.com/NVIDIA/Isaac-GR00T/tree/main/tests). These tests are run with [`pytest`](https://docs.pytest.org/en/latest/), which you can use to locally run any test modules that you've added or changed.

    For example, if you've fixed a bug in `isaac-gr00t/a/b.py`, you can run the tests specific to that module with

        pytest -v tests/a/b_test.py

    After all of the above checks have passed, you can now open [a new GitHub pull request](https://github.com/NVIDIA/Isaac-GR00T/pulls).
    Make sure you have a clear description of the problem and the solution, and include a link to relevant issues.

    We look forward to reviewing your PR!

    </details>

### Developer Certificate of Origin

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
