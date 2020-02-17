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
   
## How to manage tests

1. Use Gtest framework for testing your code
2. If your tests don't work, you have to either disable them (if they fail on all configurations) or use SKIP command for problematic configurations.
3. Each of disabled tests must have related Jira ticket, in which you have to provide detailed description of problem.
4. In the source files for each disabled test you have to specify related Jira ticket with special comment on previous line: 
	- For CVS jira tickets:
		// [Track number: S#xxxxx]
	- For VPUNND jira tickets:
		// [Track number: D#xxxxx]
5. For example:
````
// [Track number: S#12345]
TEST_F(kmbLayersTests_nightly, DISABLED_TestExportImportBlob_Convolution_After_Scale_Shift) {
    extern std::string conv_after_scale_shift;
    std::string model = conv_after_scale_shift;
    ...
````

````
// [Track number: S#67890]
TEST_P(kmbLayersTestsBias_nightly, DISABLED_TestsBias) {
    auto dim = GetParam();
    std::size_t biasesSize = 1;
    ...

````

````
// [Track number: S#45678]
INSTANTIATE_TEST_CASE_P(DISABLED_fp16_per_layer_compilation_fail, ConvolutionFP16Test,
    ::testing::ValuesIn(convolution_only_fp16), ConvolutionFP16Test::getTestCaseName);
````

````
// [Track number: S#76543]
PLUGING_CASE_WITH_PREFIX(KMB, DISABLED_, LayoutTTTest, params);
````

````
// Bad inference results.
// Compilation fails.
// [Track number: D#9876/D#3456]
TEST_F(KmbClassifyNetworkTest, DISABLED_inception_v3_tf_uint8_int8_weights_pertensor) {
    ...
````

6. Please, follow the rules, because we have python script (kmb-plugin/scripts/tests_parser/ParseTestsInfo.py) 
which parse source files and try to find related jira tickets for each of disabled tests and generate e-table
with this info in human-readable format.
7. Actual tests status for master branch can be found there (https://docs.google.com/spreadsheets/d/1wTSZ7LWGObpl7B6tQUTObNalf_mtM5C69GMf_0tB89A/edit?usp=sharing)


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
