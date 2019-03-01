#ifndef __INVERTER_H__
#define __INVERTER_H__

class Inverter {
  public:
    virtual void run() = 0;
    virtual float** get() = 0;
};

#endif
