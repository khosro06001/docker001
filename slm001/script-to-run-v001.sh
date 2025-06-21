
### with test script
docker run -it --rm slm001-test python test_local.py

### interactive
docker run -it --rm -v slm001-cache:/app/cache slm001-test
