# multi-agent-explore

by Joe Hahn,<br />
jmh.datasciences@gmail.com,<br />
10 February 2018<br />
git branch=q-learning


### Summary:
in progress...

Bitfusion Ubuntu 14 Theano???

ssh:

    ssh -i private/dl.pem ubuntu@ec2-54-200-30-59.us-west-2.compute.amazonaws.com

install:

    sudo pip2 install seaborn

clone repo:

    git clone https://github.com/joehahn/multi-agent-explore.git
    cd multi-agent-explore

get instance-id:

    echo $(ec2metadata --instance-id) > ~/instance-id
    cat ~/instance-id

browse jupyter:

    ec2-54-200-30-59.us-west-2.compute.amazonaws.com:8888

monitor GPU:

    watch -n0.1 nvidia-smi

browse tensorboard:

    ec2-54-200-30-59.us-west-2.compute.amazonaws.com:6006
