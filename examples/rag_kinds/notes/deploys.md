# Deploys

We deploy from main every weekday before 3pm. No deploys on Fridays after noon.

A deploy is rolled back with `make rollback`, which restores the previous release in about two minutes.

Database migrations run in a separate step and must be backwards compatible for one release.
