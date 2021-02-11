# Contributing

## Merge Request Process

1. Before the opening check that MR meets the requirements ([Request requirements](#request-requirements))
2. The header of the MR must be filled out according to the MR template.
3. MR must contain actual links to the CI results. [CI details](how-to-run-ci-jobs)
4. MR must be reviewed ([Review process](#review-process))
6. All discussions are resolved.

    * A discussion is closed by a developer who creates it, except there are no other explicit agreements
    * All discussion started with `Nitpick:` are not necessary and can be resolved by any developer

6. MR must be approved. The count of approves depends on changes which bring the MR
7. If the MR requires manual test runs, the test results have to be provided
   to reviewers through a MR discussion.
8. When the points above are resolved, you need to reassign the Merge Request to ([Project Maintainers](#maintainers-list))
   and ask him\her to merge. If everything is OK, the maintainer will merge MR.
9. MRs with functionality required for an upcomming release after FF milestone should be targeting release branch directly.
    * An MR with corresponding changes to master branch has to be created by the original MR author before merge to release branch.

### Request requirements
* No warnings have been introduced
* Does only one thing (Feature / Bug Fix / Optimization / Refactoring)
* Must be up-to-date with target branch
* Your MR must not contain merge commits. Next development flow required:
    * Create the feature branch based on HEAD of target branch
    * If you want to integrate changes from target branch use `git fetch` and `git rebase` commands. For example:
        * `git fetch`
        * `git rebase origin/master`
    * Resolve the conflicts and use `git push feature_branch_name -f`
* Contains list of logically separated commits. Each commit as a separate small task
    * Each commit has clear description what and why has been done
        * With Jira number when applicable
    * Request with one commit for all changes (except trivial ones) is not acceptable
        * It is recommended to do fixes for review comments in separate commits so it's easier to review
* Changes in the MR must be covered by tests
    * Checks for new functionality/behavior
    * Checks for fixed functionality/behavior
    * All required tests were actually run during validation
    * New layer must be covered in kmb-plugin per-layer tests
    * New network must be covered in kmb-plugin network tests
* For optimization additional information must be provided
    * Baseline performance
    * Optimized version performance
* Clear description for all introduced changes in header
* Link to the corresponding MR/PR if applicable
* Link to Jira ticket
* Approval from at least 2 reviewers
* Appropriate tags\labels
* Link to passed validation reports
    * IE
        * IE-MDK functional tests [IE MDK](https://wiki.ith.intel.com/display/VPUWIKI/Functional+validation+CI)
        * Nets-validation performance and accuracy check [Nets-validation](https://wiki.ith.intel.com/display/VPUWIKI/Nets-validation+CI)
    * If validation is failed due to infrastructure issues:
         * It should be reported to CI-master\Maintainer\Teams-channel
* If in the process of development or review new issues that should be fixed in a separated request
 have been found then appropriate ticket should be created

### Review process
* Request must satisfy requirements
* Request should be assigned to a reviewer first (not maintainers)
    * If you feel you don't have required expertise ask someone else to review
* Request is assigned to a project maintainer only if review has been completed and requirements have been satisfied
* Reviewers should expclicitly notify the MR author that they don't have other comments
* Request can be reviewed by any member of team, but approval is required from reviewers specified by author
* Check that request is targeted to the correct branch (e.g. master or release)
* Maintainer might perform additional code review and ask for changes in case of significant issues not covered by code reviewers

### Responsibilities

#### Request Author
* Arranging a request in according to the requirements
* Merging and passing review in time
* Immediate blockers escalating if any
* The author must take the initiative to solve the problems
* Make sure the functionality reached all the target branches (releases and master)

#### Reviewer
* Making review in according to the review process
* Reviewer is responsible for what was actually merged

#### Maintainer
* Resolving request dependencies and merging functionality in right order
* Check all requirements
* Make sure there is no potential conflicts with other changes merged recently

#### Maintainers List
* Vladislav Vinogradov
* Artemy Skrebkov
* Alexander Novak
* Sergey Losev

## How to run CI jobs
1. [IE MDK](https://wiki.ith.intel.com/display/VPUWIKI/Functional+validation+CI)
2. [Nets-validation](https://wiki.ith.intel.com/display/VPUWIKI/Nets-validation+CI)
