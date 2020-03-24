# Contributing

## Pull Request Process

1. The header of the PR must be filled out according to the questions.
2. There are CI jobs in the template which are run manually. 
   Developer must ensure PR's header contains links to actual test job.
3. All comments must be addressed and marked resolved. 
4. At least one review with "LGTM" is required, but preferably a 2nd reviewer should take a look.
5. All PR's from external contributors must be reviewed by MCM Compiler team leads
6. Changes in the PR must be covered by unit tests.
7. When the points above are resolved, you may merge the Pull Request, or
   if you do not have permission to do that, you may request the second reviewer
   to merge it for you.
8. The feature branch is to be deleted after merging
9. Additions to a compilation descriptor MUST be updated in all compilation descriptors. Don't just edit release_kmb.json.
   Add the setting (eg, a new pass) to all the other .json files. 

## CI Infrastructure 

1. The CI infrastructure is available here: https://mig-ci-jenkins.ir.intel.com/job/KeemBay/
2. Access is through AGS. Please go to https://ags.intel.com and apply for access to the group "MOVIDIUS MIG JENKINS DEVELOPER".

