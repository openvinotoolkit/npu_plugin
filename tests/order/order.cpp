#include "gtest/gtest.h"
#include "mcm/computation/tensor/shape.hpp"
#include "mcm/base/order/order.hpp"
#include "mcm/base/order/col_major.hpp"
#include "mcm/base/order/row_major.hpp"
#include "mcm/base/order/col_major_planar.hpp"
#include "mcm/base/order/row_major_planar.hpp"

TEST(order, col_major0d)
{
    mv::ColMajor order;
    mv::Shape s;

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
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
    mv::ColMajor order;
    mv::Shape s({3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
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
    mv::ColMajor order;
    mv::Shape s({2, 3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
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
    mv::ColMajor order;
    mv::Shape s({2, 3, 5});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
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
    mv::ColMajor order;
    mv::Shape s({3, 3, 5, 7});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, row_major0d)
{
    mv::RowMajor order;
    mv::Shape s;

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}


TEST(order, row_major1d)
{
    mv::RowMajor order;
    mv::Shape s({3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
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
    mv::RowMajor order;
    mv::Shape s({2, 3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
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
    mv::RowMajor order;
    mv::Shape s({2, 3, 5});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
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
    mv::RowMajor order;
    mv::Shape s({3, 3, 5, 7});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar0d)
{
    mv::ColumnMajorPlanar order;
    mv::Shape s;

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar1d)
{
    mv::ColumnMajorPlanar order;
    mv::Shape s({3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar2d)
{
    mv::ColumnMajorPlanar order;
    mv::Shape s({2, 3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar3d)
{
    mv::ColumnMajorPlanar order;
    mv::Shape s({2, 3, 5});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar4d)
{
    mv::ColumnMajorPlanar order;
    mv::Shape s({3, 3, 5, 7});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar20d)
{
    mv::RowMajorPlanar order;
    mv::Shape s;

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar21d)
{
    mv::RowMajorPlanar order;
    mv::Shape s({3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar22d)
{
    mv::RowMajorPlanar order;
    mv::Shape s({2, 3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar23d)
{
    mv::RowMajorPlanar order;
    mv::Shape s({2, 3, 5});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar24d)
{
    mv::RowMajorPlanar order;
    mv::Shape s({3, 3, 5, 7});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

