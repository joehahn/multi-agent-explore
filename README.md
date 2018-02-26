# multi-agent-explore

by Joe Hahn,<br />
jmh.datasciences@gmail.com,<br />
10 February 2018<br />
git branch=master


### Summary:
in progress...

Bitfusion Ubuntu 14 TensorFlow

ssh:

    ssh -i private/dl.pem ubuntu@ec2-34-217-26-232.us-west-2.compute.amazonaws.com

install on laptop:

    #~/miniconda2/bin/conda install -y seaborn

install on bitfusion ami:

    sudo pip install seaborn

clone repo:

    git clone https://github.com/joehahn/multi-agent-explore.git
    cd multi-agent-explore

get instance-id:

    echo $(ec2metadata --instance-id) > ~/instance-id
    cat ~/instance-id

browse jupyter:

    ec2-34-217-26-232.us-west-2.compute.amazonaws.com:8888

monitor GPU:

    watch -n0.1 nvidia-smi

browse tensorboard:

    ec2-34-217-26-232.us-west-2.compute.amazonaws.com:6006


