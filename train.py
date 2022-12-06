import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
#from util.visualizer import Visualizer
from nn_temp.dataset import *
from nn_temp.data_utils import *
import matplotlib.pyplot as plt
def main():
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    mnist_loader = dataset.dataloader
    model = create_model(opt)
    #visualizer = #Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, mnist_set in enumerate(mnist_loader):
            #iter_start_time = time.time()
            #Visualizer.reset()

            mnist = mnist_set[0]
            labels = mnist_set[1]

            mnist = mnist.unsqueeze(2).unsqueeze(2)
            hindi = random_hindi_batch_gen(Hindi_Digits().label_dict,labels=labels)
            hindi = hindi.unsqueeze(2).unsqueeze(2)
            data = {'A0': mnist[0][0], 'A1': mnist[0][1], 'A2': mnist[0][2], 'B0': hindi[0][0], 'B1': hindi[0][1], 'B2': hindi[0][2],
                    'A_paths': "", 'B_paths': ""}

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                #Visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                #t = (time.time() - iter_start_time) / opt.batchSize
                #Visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                #if opt.display_id > 0:
                    #Visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
             


        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay), time.time() - epoch_start_time)
        model.update_learning_rate()


if __name__ == '__main__':
    main()