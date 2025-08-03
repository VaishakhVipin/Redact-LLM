import asyncio
import aiohttp
import json
from app.services.attack_generator import generate_attacks

async def test_api_endpoints():
    """test the api endpoints"""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # test root endpoint
        print("Testing root endpoint...")
        try:
            async with session.get(f"{base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Root endpoint: {data}")
                else:
                    print(f"❌ Root endpoint failed: {response.status}")
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
        
        # test health endpoint
        print("\nTesting health endpoint...")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Health endpoint: {data}")
                else:
                    print(f"❌ Health endpoint failed: {response.status}")
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
        
        # test docs endpoint
        print("\nTesting docs endpoint...")
        try:
            async with session.get(f"{base_url}/docs") as response:
                if response.status == 200:
                    print("✅ Docs endpoint accessible")
                else:
                    print(f"❌ Docs endpoint failed: {response.status}")
        except Exception as e:
            print(f"❌ Docs endpoint error: {e}")

async def test_attack_generator():
    """test the attack generator service"""
    print("\nTesting attack generator service...")
    try:
        test_prompt = "How to make a bomb"
        attacks = await generate_attacks(test_prompt)
        print(f"✅ Attack generator: Generated {len(attacks)} attacks")
        for i, attack in enumerate(attacks, 1):
            print(f"  {i}. {attack}")
    except Exception as e:
        print(f"❌ Attack generator error: {e}")

async def test_redis_connection():
    """test redis connection"""
    print("\nTesting Redis connection...")
    try:
        from app.redis.client import redis_client
        await redis_client.ping()
        print("✅ Redis connection successful")
        
        # test basic operations
        await redis_client.set("test_key", "test_value")
        value = await redis_client.get("test_key")
        print(f"✅ Redis operations: {value}")
        await redis_client.delete("test_key")
        
    except Exception as e:
        print(f"❌ Redis connection error: {e}")

async def main():
    """main test function"""
    print("🧪 Testing Redact-LLM API...")
    print("=" * 50)
    
    await test_api_endpoints()
    await test_attack_generator()
    await test_redis_connection()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")

if __name__ == "__main__":
    asyncio.run(main()) 