# Contributing

## Merge Request Process

1. The header of the MR must be filled out according to the MR template.
2. There are CI jobs in the template which are run manually. 
   Developer must ensure MR's header contains links to green CI for the latest commit.  
3. There is precommit which is triggered for each MR. Precommit must be green.
4. All discussions are resolved.

    * A discussion is closed by a developer who creates it
  
5. No "thumbs down" and some "thumbs up". The count of "thumbs up" depends
   on changes which bring the MR.
6. Changes in the MR must be covered by tests.
7. If the MR requires manual test runs, the test results have to be provided
   to reviewers through a MR discussion.
8. When the points above are resolved, you may merge the Merge Request, or
   if you do not have permission to do that, you may request the second reviewer
   to merge it for you.

## How to run TeamCity Build All
1. Go to the link provided in the MR template.
2. Choose dldt branch to test (on the right side of `Build Keembay only` label)
3. Press '...' button (on the right side of 'Run' button).
4. Go to 'Parameters' tab and enter branch, you want to test, into 'reverse.dep.branch._KMB_plugin_branch'
5. Press 'Run Build' button
6. Attach a link to the job triggered in MR's description.

## How to run IE-MDK job
1. Go to the link provided in the MR template.
2. Choose dldt and kmb-plugin branches, you want to test, by entering names of the branches into `dldt_branch` and  `kmb_plugin_branch` respectively.
3. Select the following checkboxes:
    * Platforms
        * run_Ubuntu18
        * run_Yocto_ARM
    * Devices
        * run_KMB
        * run_HDDL2
3. Press `Build` button.
    * There will be a new job triggered. You can find the link to the job in `Build history`.
4. Attach a link to the job in MR's description.
