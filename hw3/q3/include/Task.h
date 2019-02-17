#ifndef __TASK_H
#define __TASK_H

class Task {

  public:
   virtual void* execute(void* args) = 0; 

};

#endif
