# Contributing to AutoPricer

Thank you for considering contributing to AutoPricer.

## Where do I go from here?

If you've noticed a bug or have a feature request, make one! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

## Fork & create a branch

If this is something you think you can fix, then fork AutoPricer and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b 325-add-new-pricing-feature
```

## Get the test suite running

Make sure you have Docker and Python installed. Follow the quickstart guide in the `README.md` to get your local environment set up. Run the test suite to ensure everything is working correctly before you start making changes:

```sh
make test
```

## Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first.

## Code formatting

We use `black` and `ruff` for code formatting and linting. Run these before committing:

```sh
make lint
```

## Make a Pull Request

At this point, you should switch back to your main branch and make sure it's up to date with AutoPricer's main branch:

```sh
git remote add upstream git@github.com:Arshiamk/autopricer.git
git checkout main
git pull upstream main
```

Then update your feature branch from your local copy of main, and push it!

```sh
git checkout 325-add-new-pricing-feature
git rebase main
git push --set-upstream origin 325-add-new-pricing-feature
```

Finally, go to GitHub and make a Pull Request!
