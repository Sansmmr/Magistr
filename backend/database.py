"""
Database client using Prisma
"""

from prisma import Prisma

prisma = Prisma()

async def get_database():
    """Get database client"""
    if not prisma.is_connected():
        await prisma.connect()
    return prisma

async def close_database():
    """Close database connection"""
    if prisma.is_connected():
        await prisma.disconnect()
