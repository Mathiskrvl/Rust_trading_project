Pour éxecuter les entrainement :

Pour l'environnement:   cargo run --bin train-encoder

Pour l'agent:   cargo run --bin train-agent

Une fois le tout entrainé lancer l'agent de trading:    cargo run



Pour entrainement dans le cloud le meilleur rapport qualité/prix est AWS G4ad, si en Spot mettre souvent des record du model dans les S3.

Le framework Burn ici utilisé permet d'entrainé nos agents avec n'importe quelle API de GPU (OpenGL, DirectX, Vulkan) grâce à l'utilisation de wgpu qui rend cela possible.

C'est un avantage certain par rapport au autre framework de deep learning classique qui sont souvent bloquer à Nvidia due a la technologie Cuda, tout en ayant une efficacité relativement égal.


Crédit:
multi-agent paper: https://arxiv.org/pdf/1706.02275.pdf \n
SAC paper : https://arxiv.org/pdf/1801.01290.pdf \n
PPO paper : https://arxiv.org/pdf/1707.06347.pdf \n
Burn framework : https://burn.dev/ \n


https://www.youtube.com/watch?v=tZTQ6S9PfkE
https://www.youtube.com/watch?v=ioidsRlf79o&t=929s
https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/sac.py#L326
https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/ppo_continuous2.py
