#---------------------------------------------
# the input type to model is event tensor 
#---------------------------------------------

class model_abc(): 
    def forward(self, event_tensor, prev_states):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return img_pred, states
      
#-------------------------------------------
# example to call model with event tensor
#-------------------------------------------
reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)    

def update_reconstruction(self, event_tensor, event_tensor_id, stamp=None):
    with torch.no_grad():
          events = event_tensor.unsqueeze(dim=0)
          events = events.to(self.device)

          events = self.event_preprocessor(events)

          # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
          events_for_each_channel = {'grayscale': self.crop.pad(events)}
          reconstructions_for_each_channel = {}
          if self.perform_color_reconstruction:
              events_for_each_channel['R'] = self.crop_halfres.pad(events[:, :, 0::2, 0::2])
              events_for_each_channel['G'] = self.crop_halfres.pad(events[:, :, 0::2, 1::2])
              events_for_each_channel['W'] = self.crop_halfres.pad(events[:, :, 1::2, 0::2])
              events_for_each_channel['B'] = self.crop_halfres.pad(events[:, :, 1::2, 1::2])

           # Reconstruct new intensity image for each channel (grayscale + RGBW if color reconstruction is enabled)
           for channel in events_for_each_channel.keys():
              #with CudaTimer('Inference'):
              new_predicted_frame, states = self.model(events_for_each_channel[channel], self.last_states_for_each_channel[channel])
                # events_for_each_channel[channel] is a Tensor type, (1, 5, 184, 240)
