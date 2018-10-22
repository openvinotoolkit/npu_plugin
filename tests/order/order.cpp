#include "gtest/gtest.h"
#include "include/mcm/order/order.hpp"

TEST(order, row_major1d)
{
    mv::Shape s({1});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, row_major2d)
{
    mv::Shape s({2, 2});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, row_major3d)
{
    mv::Shape s({3, 3, 3});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, row_major4d)
{
    mv::Shape s({3, 3, 3, 3});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, row_major5d)
{
    mv::Shape s({3, 3, 3, 3, 3});
    mv::Order order(mv::Order::getRowMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, col_major1d)
{
    mv::Shape s({1});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, col_major2d)
{
    mv::Shape s({2, 2});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, col_major3d)
{
    mv::Shape s({3, 3, 3});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, col_major4d)
{
    mv::Shape s({3, 3, 3, 3});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, col_major5d)
{
    mv::Shape s({3, 3, 3, 3, 3});
    mv::Order order(mv::Order::getColMajorID(s.ndims()));

    try
    {
        for(unsigned i = 0; i < order.size(); ++i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;


        for(int i = order.size() - 1; i >= 0; --i)
        {
            std::cout << order[i];
            if(order.isFirstContiguousDimensionIndex(i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

//TODO: Order tests must be rewritten using the above template
