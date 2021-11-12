## Summary

(Please add a short summary why your merge request is useful.)

## Related MRs

(Please add links to related MRs (if you have such MRs) and a small note why you depend on it.)

* <mr-link> (<description>)

## Related tickets

(Please list tickets which the MR closes if you have any.)

* [EISW-XXXXX](https://jira.devtools.intel.com/browse/EISW-XXXXX)

## Reviewers

(Please list reviewers of the MR here.)

* [ ] <@mention>

## CI

(Please replace the links below with your own.)

#### Mandatory validation

(Default filter: `*precommit*:*smoke*`. Empty functional_tests filter for any major changes.)

* [ ] https://dsp-ci-icv.inn.intel.com/job/IE-MDK/job/manual/job/Ubuntu-Yocto/build
* [ ] https://dsp-ci-icv.inn.intel.com/job/IE-MDK/job/manual/job/Windows_dKMB/build

#### Validation for compiler changes / performance affected

(`*MLIR/precommit*` nets_included filter for VPUX compiler, `*MCM/precommit*` for MCM compiler.)

* [ ] https://dsp-ci-icv.inn.intel.com/job/Nets-Validation/job/manual/job/Yocto/build

#### Validation for dKMB focused changes in compiler or major changes

(Filters are the same as for Yocto.)

* [ ] https://dsp-ci-icv.inn.intel.com/job/Nets-Validation/job/manual/job/Windows/build

#### Compilation and single-image test Validation on moviSim for MTL related changes

(Default filter: `*MTL*`)

* [ ] https://dsp-ci-icv.inn.intel.com/job/Nets-Validation/job/manual/job/MoviSim/build
