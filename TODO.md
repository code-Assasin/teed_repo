<!-- add checklist -->
- [x] Input and output need to be 3 channel. 
- [ ] Integrate with original repo.
- [ ] clean up original repo comments especially the dimensions in the teed model.
- [ ] Add gaussian noise corruption from the fast corruption code but of moderate strength ? 
- [ ] Rewrite loss function to be more readable and as a class so the functions work for 3 channels. and on gpu
- [ ] Clean up wandb pics so they all appear in one row and not multiple cells.
- [ ] Might have to change architecture because currently pred combined doesn't have any useful information and its not learning either.Also may have to condition on the input image.? 
- [ ] Add run name and save name functions.
