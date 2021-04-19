# Contributing

## Merge Request Process

1. Open "WIP" MR if functionality is not quite ready yet but you want others to take a look or it is a POC and you want to kick off an architectural discussion.
1. To prepare an MR for review: remove WIP status and make sure it meets following [requirements](#request-requirements).

1. MR must be reviewed according to the [review process](#review-process).
    * Start the review process with people experienced in this particular area and/or involved into architectural debates.
    * Explicitly ask for review (in MR header and through other communication channels).
    * Make sure to rebase the branch and restart validation after addressing all the comments.
    * MR's author is [responsible](#responsibilities) for the review to be passed in time.

1. Assign the MR to a [project maintainer](#maintainers-list) only if you have at least 2 approves.
    * A maintainer makes sure all the [requirements](#request-requirements) are satisfied and presses the merge button.
    * MR's author is [responsible](#responsibilities) for MR to be merged in time.
    * MR's author is [responsible](#responsibilities) for merging to a proper target branch for a proper milestone.

## Request requirements

1. MR title must start with a jira ticket number in the following format: "EISW-####:Fix for this and that".
    * Jira ticket must include the problem statement, acceptance criteria, links to design/architecture docs if any.
1. MR header must be filled out according to the MR template.
    * The header must contain clear description and the target milestone.
    * Links to connected\depended MRs and PRs if any.
    * Links to passed [validation](how-to-run-ci-jobs) reports:
        * IE-MDK functional tests [IE MDK](https://wiki.ith.intel.com/display/VPUWIKI/Functional+validation+CI).
        * Nets-validation performance and accuracy check [Nets-validation](https://wiki.ith.intel.com/display/VPUWIKI/Nets-validation+CI).
        * If validation is failed due to infrastructure issues it should be reported to CI-master\Maintainer\Teams-channel.
1. The MR does only one thing (Feature / Bug Fix / Optimization / Refactoring).
1. All the changes are accompanied with clear comments incorporated into the source code.
1. The feature branch is up to date with the target branch.
1. The MR is targeting the right target branch for the right milestone.
1. Your MR must not contain merge commits. Next development flow required:
    * Create the feature branch based on HEAD of target branch.
    * If you want to integrate changes from target branch use `git fetch` and `git rebase` commands. For example:
        * `git fetch`
        * `git rebase origin/master`
        * Then resolve the conflicts and use `git push feature_branch_name -f`
1. Contains list of logically separated commits. Each commit as a separate small task.
    * Each commit has clear description what and why has been done.
        * With Jira number when applicable.
    * Request with one commit for all changes (except trivial ones) is not acceptable.
        * It is recommended to do fixes for review comments in separate commits so it's easier to review.
1. No warnings have been introduced.
1. Changes in the MR must be covered by tests.
    * Checks for new functionality/behavior.
    * Checks for fixed functionality/behavior.
    * All required tests were actually run during validation.
    * New layer must be covered in kmb-plugin per-layer tests.
    * New network must be covered in kmb-plugin network tests.
1. For optimization additional information must be provided.
    * Baseline performance.
    * Optimized version performance.
1. If issues identified during the review process for some reason cannot be addressed in the same MR, all the created tickets and corresponding agreements must be included into the MR header.

## Review process
1. MR must be approved. The count of approves depends on changes which bring the MR.
1. If the MR requires manual test runs, the test results have to be provided to reviewers through a MR discussion.
1. Request should be assigned to a reviewer first (not maintainer).
    * If you feel you don't have required expertise ask someone else to review.
1. Request is assigned to a project maintainer only if review has been completed and all the review comments addressed.
1. Reviewers should expclicitly notify the MR author that they don't have other comments.
1. Request can be reviewed by any member of team, but approval is required from reviewers specified by author.
1. Check that request is targeted to the correct branch (e.g. master or release).
1. All discussions must be resolved.
    1. A discussion is closed by a developer who creates it, except there are no other explicit agreements.
    1. All discussion started with `Nitpick:` are not necessary and can be resolved by any developer.
1. Maintainer might perform additional code review and ask for changes in case of significant issues not covered by code reviewers.

## Responsibilities

### Request Author
* Arranging a request in according to the requirements.
* Merging and passing review in time.
* Immediate blockers escalating if any.
* The author must take the initiative to solve the problems.
* Make sure the functionality reached all the target branches.
* Resolve potential conflicts if requested by a maintainer during a merge of release branch into master.

### Reviewer
* Making review in according to the review process.
* Reviewer is responsible for what was actually merged.

### Maintainer
* Resolving request dependencies and merging functionality in right order.
* Check all requirements.
* Make sure there is no potential conflicts with other changes merged recently.

### Maintainers List
* VPUX Plugin - Artemy, Skrebkov
* mcmCompiler - Marina, Mineeva
* MLIR Compiler - Vlad, Vinogradov

## How to run CI jobs
1. [IE MDK](https://wiki.ith.intel.com/display/VPUWIKI/Functional+validation+CI)
2. [Nets-validation](https://wiki.ith.intel.com/display/VPUWIKI/Nets-validation+CI)
