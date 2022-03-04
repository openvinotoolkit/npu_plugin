// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <iostream>

#define checkbuf(val) if (!(val)) {    \
                std::cout << "Can't new " << #val << std::endl;    \
                exit(0);}

template<class T>
class CThreadQueue {
 public:
    explicit CThreadQueue(int sz = 0x7FFFFFFF) :
        _size_limit(sz), _max_size(0), _closed(false) {
        _m = new std::mutex();
        checkbuf(_m);
        _cv = new std::condition_variable();
        checkbuf(_cv);
        _cv_notfull = new std::condition_variable();
        checkbuf(_cv_notfull);
    }
    ~CThreadQueue() {
        if (_m) {
            delete _m;
            _m = nullptr;
        }
        if (_cv) {
            delete _cv;
            _cv = nullptr;
        }
        if (_cv_notfull) {
            delete _cv_notfull;
            _cv_notfull = nullptr;
        }
    }

    // with Filter on element(get specific element)
    template<class FilterFunc>
    bool get(T &ret, FilterFunc filter) {
        std::unique_lock<std::mutex> lk(*_m);

        typename std::deque<T>::iterator it;
        (*_cv).wait(lk, [this, filter, &it] {
          // mutex auto relocked
          // check from the First Input to meet FIFO requirement
          for (it = _q.begin(); it != _q.end(); ++it)
              if (filter(*it)) return true;

          // if closed & not-found, then we will never found it in the future
          if (_closed) return true;

          return false;
        });

        if (static_cast<int>(_q.size()) > _max_size) {
            _max_size = _q.size();
        }

        // nothing will found in the future
        if (it == _q.end() || _closed) {
            return false;
        }

        ret = *it;        // copy construct the return value
        _q.erase(it);    // remove from deque

        if (_q.size() < _size_limit) {
            (*_cv_notfull).notify_all();
        }

        return true;
    }

    bool get(T &ret) {
        return get(ret, [](const T &) { return true; });
    }

    void put(const T &obj) {
        std::unique_lock<std::mutex> lk(*_m);

        if (!_closed) {
            (*_cv_notfull).wait(lk, [this] {
              return (_q.size() < _size_limit) || _closed;
            });

            if (!_closed)
                _q.push_back(obj);
        }

        (*_cv).notify_all();
    }

    void close(void) {
        std::unique_lock<std::mutex> lk(*_m);
        _closed = true;
        (*_cv).notify_all();
    }
    int size(void) {
        std::unique_lock<std::mutex> lk(*_m);
        return _q.size();
    }

 private:
    size_t _size_limit;
    std::deque<T> _q;
    std::mutex *_m = nullptr;
    std::condition_variable *_cv = nullptr;
    std::condition_variable *_cv_notfull = nullptr;
    int _max_size;
    bool _closed;
};
