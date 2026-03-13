before transfer:

```
Train | loss=0.2608, micro_f1=0.8081, precision=0.7609, recall=0.8616, label_acc=0.8851, exact_match=0.5297     
Val[0] | loss=0.0923, micro_f1=0.9432, precision=0.9529, recall=0.9337, label_acc=0.9682, exact_match=0.8231    
Val[1] | loss=0.6505, micro_f1=0.4092, precision=0.4290, recall=0.3912, label_acc=0.7437, exact_match=0.1000    
Saved new best checkpoint to: ./checkpoints/before_transfer/best_model.pt                               
Epoch [1/10] - 19.5s                                                                                            
Train | loss=0.0761, micro_f1=0.9561, precision=0.9602, recall=0.9521, label_acc=0.9755, exact_match=0.8609     
Val[0] | loss=0.1150, micro_f1=0.9191, precision=0.9309, recall=0.9077, label_acc=0.9548, exact_match=0.7894    
Val[1] | loss=0.8697, micro_f1=0.3435, precision=0.3454, recall=0.3416, label_acc=0.7037, exact_match=0.0450    

Epoch [2/10] - 17.6s                                                                                            
Train | loss=0.0412, micro_f1=0.9808, precision=0.9844, recall=0.9773, label_acc=0.9893, exact_match=0.9297     
Val[0] | loss=0.1010, micro_f1=0.9337, precision=0.9433, recall=0.9243, label_acc=0.9629, exact_match=0.8269    
Val[1] | loss=0.7698, micro_f1=0.3678, precision=0.3844, recall=0.3526, label_acc=0.7250, exact_match=0.0600    
         
Epoch [3/10] - 15.7s                                                                                            
Train | loss=0.0303, micro_f1=0.9867, precision=0.9880, recall=0.9855, label_acc=0.9926, exact_match=0.9528     
Val[0] | loss=0.0548, micro_f1=0.9660, precision=0.9694, recall=0.9627, label_acc=0.9809, exact_match=0.8988    
Val[1] | loss=0.7867, micro_f1=0.3898, precision=0.4000, recall=0.3802, label_acc=0.7300, exact_match=0.0950    
Saved new best checkpoint to: ./checkpoints/before_transfer/best_model.pt                                                
Epoch [4/10] - 16.7s
Train | loss=0.0208, micro_f1=0.9919, precision=0.9936, recall=0.9903, label_acc=0.9955, exact_match=0.9703
Val[0] | loss=0.0609, micro_f1=0.9608, precision=0.9696, recall=0.9522, label_acc=0.9780, exact_match=0.8794
Val[1] | loss=0.9131, micro_f1=0.3880, precision=0.4024, recall=0.3747, label_acc=0.7319, exact_match=0.1300

Epoch [5/10] - 18.0s
Train | loss=0.0148, micro_f1=0.9953, precision=0.9957, recall=0.9949, label_acc=0.9974, exact_match=0.9837
Val[0] | loss=0.0561, micro_f1=0.9667, precision=0.9721, recall=0.9613, label_acc=0.9812, exact_match=0.8963
Val[1] | loss=0.9029, micro_f1=0.3818, precision=0.3953, recall=0.3691, label_acc=0.7287, exact_match=0.0800

Epoch [6/10] - 16.3s
Train | loss=0.0136, micro_f1=0.9954, precision=0.9964, recall=0.9943, label_acc=0.9974, exact_match=0.9842
Val[0] | loss=0.0619, micro_f1=0.9627, precision=0.9737, recall=0.9519, label_acc=0.9791, exact_match=0.8837
Val[1] | loss=0.9379, micro_f1=0.3712, precision=0.3886, recall=0.3554, label_acc=0.7269, exact_match=0.0650

Epoch [7/10] - 17.5s
Train | loss=0.0112, micro_f1=0.9962, precision=0.9968, recall=0.9956, label_acc=0.9979, exact_match=0.9875
Val[0] | loss=0.0700, micro_f1=0.9643, precision=0.9781, recall=0.9508, label_acc=0.9801, exact_match=0.8825
Val[1] | loss=1.1536, micro_f1=0.3470, precision=0.3555, recall=0.3388, label_acc=0.7106, exact_match=0.0800

Epoch [8/10] - 16.6s
Train | loss=0.0092, micro_f1=0.9975, precision=0.9976, recall=0.9974, label_acc=0.9986, exact_match=0.9923
Val[0] | loss=0.0503, micro_f1=0.9696, precision=0.9741, recall=0.9652, label_acc=0.9829, exact_match=0.9013
Val[1] | loss=1.0956, micro_f1=0.3561, precision=0.3687, recall=0.3444, label_acc=0.7175, exact_match=0.0400

Epoch [9/10] - 33.6s
Train | loss=0.0083, micro_f1=0.9973, precision=0.9974, recall=0.9972, label_acc=0.9985, exact_match=0.9911
Val[0] | loss=0.0541, micro_f1=0.9683, precision=0.9711, recall=0.9655, label_acc=0.9821, exact_match=0.8975
Val[1] | loss=1.0516, micro_f1=0.3703, precision=0.3932, recall=0.3499, label_acc=0.7300, exact_match=0.0750

Epoch [10/10] - 19.9s

Training finished.
Best val micro-F1: 0.6779
```



After Transfer:

```
Loading checkpoint from ./checkpoints/before_transfer/last_model.pt                      
Checkpoint loaded.                                                                                              
Train | loss=0.4384, micro_f1=0.6841, precision=0.7021, recall=0.6671, label_acc=0.8569, exact_match=0.3300     
Val[0] | loss=0.5979, micro_f1=0.6885, precision=0.7622, recall=0.6278, label_acc=0.8394, exact_match=0.3162    
Val[1] | loss=0.1714, micro_f1=0.8709, precision=0.8977, recall=0.8457, label_acc=0.9431, exact_match=0.6250    
Saved new best checkpoint to: ./checkpoints/after_transfer/best_model.pt                              
Epoch [1/10] - 23.6s                                                                                            
Train | loss=0.0997, micro_f1=0.9372, precision=0.9465, recall=0.9280, label_acc=0.9711, exact_match=0.8025     
Val[0] | loss=0.6798, micro_f1=0.6444, precision=0.7444, recall=0.5681, label_acc=0.8227, exact_match=0.2419    
Val[1] | loss=0.0889, micro_f1=0.9465, precision=0.9426, recall=0.9504, label_acc=0.9756, exact_match=0.8450    
Saved new best checkpoint to: ./checkpoints/after_transfer/best_model.pt                                      
Epoch [2/10] - 41.5s                                                                                            
Train | loss=0.0494, micro_f1=0.9852, precision=0.9833, recall=0.9872, label_acc=0.9931, exact_match=0.9475     
Val[0] | loss=0.6185, micro_f1=0.6344, precision=0.7344, recall=0.5584, label_acc=0.8180, exact_match=0.2531    
Val[1] | loss=0.0567, micro_f1=0.9672, precision=0.9593, recall=0.9752, label_acc=0.9850, exact_match=0.9000    
Saved new best checkpoint to: ./checkpoints/after_transfer/best_model.pt                                
Epoch [3/10] - 22.4s                                                                                            
Train | loss=0.0334, micro_f1=0.9899, precision=0.9893, recall=0.9906, label_acc=0.9953, exact_match=0.9663     
Val[0] | loss=0.5329, micro_f1=0.6566, precision=0.7712, recall=0.5717, label_acc=0.8309, exact_match=0.2631    
Val[1] | loss=0.0426, micro_f1=0.9781, precision=0.9702, recall=0.9862, label_acc=0.9900, exact_match=0.9250    
Saved new best checkpoint to: ./checkpoints/after_transfer/best_model.pt

Epoch [4/10] - 23.5s
Train | loss=0.0255, micro_f1=0.9926, precision=0.9900, recall=0.9953, label_acc=0.9966, exact_match=0.9762
Val[0] | loss=0.5615, micro_f1=0.6320, precision=0.7570, recall=0.5424, label_acc=0.8214, exact_match=0.2350
Val[1] | loss=0.0376, micro_f1=0.9781, precision=0.9702, recall=0.9862, label_acc=0.9900, exact_match=0.9250

Epoch [5/10] - 22.9s
Train | loss=0.0213, micro_f1=0.9923, precision=0.9893, recall=0.9953, label_acc=0.9964, exact_match=0.9812
Val[0] | loss=0.5943, micro_f1=0.6151, precision=0.7460, recall=0.5233, label_acc=0.8148, exact_match=0.2319
Val[1] | loss=0.0370, micro_f1=0.9767, precision=0.9701, recall=0.9835, label_acc=0.9894, exact_match=0.9250

Epoch [6/10] - 20.8s
Train | loss=0.0180, micro_f1=0.9926, precision=0.9919, recall=0.9933, label_acc=0.9966, exact_match=0.9775
Val[0] | loss=0.6216, micro_f1=0.5991, precision=0.7457, recall=0.5007, label_acc=0.8105, exact_match=0.2006
Val[1] | loss=0.0346, micro_f1=0.9795, precision=0.9728, recall=0.9862, label_acc=0.9906, exact_match=0.9350

Epoch [7/10] - 20.5s
Train | loss=0.0144, micro_f1=0.9950, precision=0.9946, recall=0.9953, label_acc=0.9977, exact_match=0.9812
Val[0] | loss=0.6264, micro_f1=0.5986, precision=0.7587, recall=0.4943, label_acc=0.8126, exact_match=0.1912
Val[1] | loss=0.0325, micro_f1=0.9753, precision=0.9726, recall=0.9780, label_acc=0.9887, exact_match=0.9150

Epoch [8/10] - 20.5s
Train | loss=0.0141, micro_f1=0.9946, precision=0.9940, recall=0.9953, label_acc=0.9975, exact_match=0.9812
Val[0] | loss=0.6505, micro_f1=0.5784, precision=0.7353, recall=0.4767, label_acc=0.8035, exact_match=0.1856
Val[1] | loss=0.0341, micro_f1=0.9795, precision=0.9728, recall=0.9862, label_acc=0.9906, exact_match=0.9350

Epoch [9/10] - 20.7s
Train | loss=0.0112, micro_f1=0.9956, precision=0.9940, recall=0.9973, label_acc=0.9980, exact_match=0.9850
Val[0] | loss=0.6258, micro_f1=0.5875, precision=0.7510, recall=0.4825, label_acc=0.8084, exact_match=0.2000
Val[1] | loss=0.0332, micro_f1=0.9767, precision=0.9727, recall=0.9807, label_acc=0.9894, exact_match=0.9200

Epoch [10/10] - 23.1s

Training finished.
Best val micro-F1: 0.8174
```



Mix:

```
Train | loss=0.2629, micro_f1=0.8031, precision=0.7624, recall=0.8483, label_acc=0.8854, exact_match=0.5253     
Val[0] | loss=0.1124, micro_f1=0.9236, precision=0.9307, recall=0.9166, label_acc=0.9571, exact_match=0.7756    
Val[1] | loss=0.1419, micro_f1=0.9177, precision=0.9294, recall=0.9063, label_acc=0.9631, exact_match=0.7550    
Saved new best checkpoint to: ./checkpoints/mix_training/best_model.pt                                    
Epoch [1/10] - 42.4s                                                                                            
Train | loss=0.0771, micro_f1=0.9554, precision=0.9612, recall=0.9496, label_acc=0.9756, exact_match=0.8546     
Val[0] | loss=0.0913, micro_f1=0.9410, precision=0.9494, recall=0.9329, label_acc=0.9670, exact_match=0.8244    
Val[1] | loss=0.0992, micro_f1=0.9140, precision=0.9364, recall=0.8926, label_acc=0.9619, exact_match=0.7400    
Saved new best checkpoint to: ./checkpoints/mix_training/best_model.pt

Epoch [2/10] - 29.1s                                                                                            
Train | loss=0.0470, micro_f1=0.9756, precision=0.9784, recall=0.9727, label_acc=0.9866, exact_match=0.9153     
Val[0] | loss=0.0699, micro_f1=0.9551, precision=0.9601, recall=0.9503, label_acc=0.9748, exact_match=0.8544    
Val[1] | loss=0.0582, micro_f1=0.9696, precision=0.9723, recall=0.9669, label_acc=0.9862, exact_match=0.9000    
Saved new best checkpoint to: ./checkpoints/mix_training/best_model.pt                                
Epoch [3/10] - 43.5s                                                                                            
Train | loss=0.0270, micro_f1=0.9895, precision=0.9910, recall=0.9881, label_acc=0.9942, exact_match=0.9603     
Val[0] | loss=0.0653, micro_f1=0.9595, precision=0.9656, recall=0.9536, label_acc=0.9773, exact_match=0.8775    
Val[1] | loss=0.0660, micro_f1=0.9517, precision=0.9530, recall=0.9504, label_acc=0.9781, exact_match=0.8550    
                     
Epoch [4/10] - 29.5s                                                                                            
Train | loss=0.0205, micro_f1=0.9920, precision=0.9921, recall=0.9919, label_acc=0.9956, exact_match=0.9704     
Val[0] | loss=0.0554, micro_f1=0.9665, precision=0.9771, recall=0.9561, label_acc=0.9812, exact_match=0.8919
Val[1] | loss=0.0534, micro_f1=0.9588, precision=0.9562, recall=0.9614, label_acc=0.9812, exact_match=0.8950
Saved new best checkpoint to: ./checkpoints/mix_training/best_model.pt

Epoch [5/10] - 27.7s
Train | loss=0.0141, micro_f1=0.9961, precision=0.9967, recall=0.9955, label_acc=0.9979, exact_match=0.9857
Val[0] | loss=0.0565, micro_f1=0.9657, precision=0.9793, recall=0.9525, label_acc=0.9809, exact_match=0.8887
Val[1] | loss=0.0469, micro_f1=0.9629, precision=0.9615, recall=0.9642, label_acc=0.9831, exact_match=0.8950
Saved new best checkpoint to: ./checkpoints/mix_training/best_model.pt

Epoch [6/10] - 28.0s
Train | loss=0.0142, micro_f1=0.9952, precision=0.9957, recall=0.9946, label_acc=0.9973, exact_match=0.9836
Val[0] | loss=0.0565, micro_f1=0.9650, precision=0.9749, recall=0.9552, label_acc=0.9804, exact_match=0.8900
Val[1] | loss=0.0690, micro_f1=0.9474, precision=0.9526, recall=0.9421, label_acc=0.9762, exact_match=0.8650

Epoch [7/10] - 35.6s
Train | loss=0.0136, micro_f1=0.9946, precision=0.9945, recall=0.9948, label_acc=0.9970, exact_match=0.9822
Val[0] | loss=0.0618, micro_f1=0.9622, precision=0.9689, recall=0.9555, label_acc=0.9787, exact_match=0.8862
Val[1] | loss=0.0557, micro_f1=0.9599, precision=0.9639, recall=0.9559, label_acc=0.9819, exact_match=0.8950

Epoch [8/10] - 30.6s
Train | loss=0.0092, micro_f1=0.9968, precision=0.9972, recall=0.9965, label_acc=0.9982, exact_match=0.9883
Val[0] | loss=0.0621, micro_f1=0.9670, precision=0.9780, recall=0.9563, label_acc=0.9816, exact_match=0.8912
Val[1] | loss=0.0407, micro_f1=0.9673, precision=0.9569, recall=0.9780, label_acc=0.9850, exact_match=0.9100
Saved new best checkpoint to: ./checkpoints/mix_training/best_model.pt

Epoch [9/10] - 28.4s
Train | loss=0.0087, micro_f1=0.9967, precision=0.9968, recall=0.9966, label_acc=0.9982, exact_match=0.9876
Val[0] | loss=0.0886, micro_f1=0.9475, precision=0.9551, recall=0.9400, label_acc=0.9705, exact_match=0.8531
Val[1] | loss=0.0536, micro_f1=0.9633, precision=0.9516, recall=0.9752, label_acc=0.9831, exact_match=0.8950

Epoch [10/10] - 33.8s

Training finished.
Best val micro-F1: 0.9672
```

