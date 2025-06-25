docker run -d ubuntu bash -c "shuf -i 1-10000 -n 1 -o /data.txt && tail -f /dev/null"

# In case you're curious about the command, we're starting a bash shell
# and invoking two commands (why we have the &&). The first portion
# picks a single random number and writes it to /data.txt. The second
# command is simply watching a file to keep the container running.

