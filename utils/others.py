import wandb

def wandb_init(args):
    if(args.log == 'wandb'):
        wandb.init(project='ESAM', entity = 'parroteffect', name = args.name)

def wandb_log(args, name, v):
    if(args.log == 'wandb'):
        wandb.log({
            name : v,
        })

        
def wandb_log_some(args, names, vs):
    if(args.log == 'wandb'):
        wandb.log({
            names[0] : vs[0],
            names[1] : vs[1],
            names[2] : vs[2],
            names[3] : vs[3],
        })
    else:
        print(vs)