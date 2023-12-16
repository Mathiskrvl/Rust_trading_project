Pour éxecuter les entrainement :

Pour l'environnement:   cargo run --bin train-encoder

Pour l'agent:   cargo run --bin train-agent

Une fois le tout entrainé lancer l'agent de trading:    cargo run



Pour entrainement dans le cloud le meilleur rapport qualité/prix est AWS G4ad, si en Spot mettre souvent des record du model dans les S3.

Le framework Burn ici utilisé permet d'entrainé nos agents avec n'importe quelle API de GPU (OpenGL, DirectX, Vulkan) grâce à l'utilisation de wgpu qui rend cela possible.

C'est un avantage certain par rapport au autre framework de deep learning classique qui sont souvent bloquer à Nvidia due a la technologie Cuda, tout en ayant une efficacité relativement égal.
